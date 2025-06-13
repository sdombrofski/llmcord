import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Any, Literal, Optional
import json
import re

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude", "gemini", "gemma", "llama", "pixtral", "mistral", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 3

MAX_MESSAGE_NODES = 500
DISCORD_MAX_MESSAGE_LENGTH = 2000  # Discord's maximum message content length (plain text)
DISCORD_MAX_EMBED_DESCRIPTION = 4096  # Discord's maximum embed description length


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices


@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

    await discord_bot.tree.sync()


def convert_messages_for_ollama(messages):
    """Convert OpenAI format messages to Ollama format"""
    ollama_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            # Ollama doesn't have system role, we'll prepend to first user message
            continue
            
        # Handle content that could be string or list (for images)
        content = msg["content"]
        if isinstance(content, list):
            # Extract text from complex content
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
            content = "\n".join(text_parts) if text_parts else ""
        
        ollama_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    return ollama_messages


def remove_thinking_sections(text):
    """Remove thinking sections from model responses"""
    # Remove content between <thinking> and </thinking> tags
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove content between <think> and </think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove "Let me think..." style thinking patterns
    text = re.sub(r'^(Let me think.*?\n\n)', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(I need to think.*?\n\n)', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(Thinking.*?\n\n)', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove reasoning chains that start with "First," "Step 1:", etc.
    # but only if they appear at the beginning and are followed by actual response
    lines = text.split('\n')
    cleaned_lines = []
    skip_reasoning = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line starts a reasoning section
        if any(line_lower.startswith(pattern) for pattern in [
            'first,', 'step 1:', 'let me analyze', 'let me break', 
            'to answer this', 'i should consider', 'thinking through'
        ]):
            skip_reasoning = True
            continue
            
        # Check if we've reached the actual answer
        if skip_reasoning and any(line_lower.startswith(pattern) for pattern in [
            'the answer', 'in conclusion', 'therefore', 'so,', 'thus,', 
            'my response', 'to summarize', 'simply put'
        ]):
            skip_reasoning = False
            cleaned_lines.append(line)
            continue
            
        if not skip_reasoning:
            cleaned_lines.append(line)
    
    # Clean up extra whitespace
    result = '\n'.join(cleaned_lines).strip()
    
    # Remove multiple consecutive newlines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or config["allow_dms"] if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.split("/", 1)
    model_parameters = config["models"].get(provider_slash_model, None)

    # Hardcode Ollama host URL
    if provider == "ollama":
        base_url = "http://0.0.0.0:11434"
    else:
        # Get base URL without /v1 for direct Ollama API
        base_url = config["providers"][provider]["base_url"].replace("/v1", "")

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config["max_text"]
    max_images = config["max_images"] if accept_images else 0
    max_messages = config["max_messages"]

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Prepare system prompt
    system_prompt = ""
    if config.get("system_prompt"):
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
        system_prompt = "\n".join([config["system_prompt"]] + system_prompt_extras)

    # Convert messages for Ollama
    ollama_messages = convert_messages_for_ollama(messages[::-1])

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = ""
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    use_plain_responses = config["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    try:
        async with new_msg.channel.typing():
            # Prepare Ollama chat request
            payload = {
                "model": model,
                "messages": ollama_messages,
                "stream": True
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if model_parameters:
                payload.update(model_parameters)

            edit_task = None
            
            async with httpx_client.stream('POST', f"{base_url}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        if data.get("done", False):
                            break
                            
                        if "message" in data and "content" in data["message"]:
                            chunk_content = data["message"]["content"]
                            curr_content += chunk_content

                            if response_contents == [] or len(response_contents[-1] + curr_content) > max_message_length:
                                response_contents.append("")

                            response_contents[-1] = curr_content

                            if not use_plain_responses:
                                ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                                is_final_edit = data.get("done", False)
                                # Only update if we have enough content or it's the final edit
                                has_enough_content = len(curr_content) >= 50  # Wait for at least 50 characters
                                
                                if (len(response_contents) == 1 or (ready_to_edit and has_enough_content) or is_final_edit):
                                    if edit_task != None:
                                        try:
                                            await edit_task
                                        except discord.HTTPException:
                                            # If we hit rate limit, just continue
                                            pass

                                    # Remove thinking sections before displaying
                                    display_content = remove_thinking_sections(curr_content) if is_final_edit else curr_content
                                    
                                    # Truncate content to fit Discord's limits with larger buffer
                                    truncated_content = display_content[:DISCORD_MAX_EMBED_DESCRIPTION-len(STREAMING_INDICATOR)-50] if not is_final_edit else display_content[:DISCORD_MAX_EMBED_DESCRIPTION-50]
                                    if len(display_content) > len(truncated_content):
                                        truncated_content += "..." if is_final_edit else "..."
                                    
                                    embed.description = truncated_content if is_final_edit else (truncated_content + STREAMING_INDICATOR)
                                    embed.color = EMBED_COLOR_COMPLETE if is_final_edit else EMBED_COLOR_INCOMPLETE

                                    if len(response_msgs) == 0:
                                        response_msg = await new_msg.reply(embed=embed, silent=True)
                                        response_msgs.append(response_msg)
                                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                        await msg_nodes[response_msg.id].lock.acquire()
                                    else:
                                        try:
                                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                        except discord.HTTPException:
                                            # If we hit rate limit, skip this update
                                            pass

                                    last_task_time = dt.now().timestamp()
                                    
                    except json.JSONDecodeError:
                        continue

            if use_plain_responses:
                reply_to_msg = new_msg
                # Remove thinking sections before sending
                cleaned_content = remove_thinking_sections(curr_content)
                # Truncate content to fit Discord's 2000 character limit with buffer
                truncated_content = cleaned_content[:DISCORD_MAX_MESSAGE_LENGTH-50]  # Increased buffer
                if len(cleaned_content) > len(truncated_content):
                    truncated_content += "..."
                response_msg = await reply_to_msg.reply(content=truncated_content, suppress_embeds=True)
                response_msgs.append(response_msg)
                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    # Clean the final content before storing
    final_cleaned_content = remove_thinking_sections(curr_content)
    
    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = final_cleaned_content
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


asyncio.run(main())
