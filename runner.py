import aiohttp
import argparse
from typing import Optional
from dataclasses import dataclass

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

@dataclass
class AgentConfig:
    url: str
    apikey: str
    speed: str = "normal"
    emotion: list = None
    prompt: str = ""
    voice_id: str = ""
    session_time: float = 3600  # Default 1 hour
    token: str = None

def create_parser():
    parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument("-u", "--url", type=str, required=True, help="URL of the Daily room to join")
    parser.add_argument("-k", "--apikey", type=str, required=True, help="Daily API Key")
    parser.add_argument("--speed", type=str, default="normal", help="Voice speed")
    parser.add_argument("--emotion", type=str, help="Comma-separated list of emotions")
    parser.add_argument("--prompt", type=str, required=True, help="System prompt for the bot")
    parser.add_argument("--voice-id", type=str, required=True, help="Voice ID for TTS")
    parser.add_argument("--session-time", type=float, default=3600, help="Session expiry time in seconds")
    parser.add_argument("-t", "--token", type=str, help="Optional: Direct token for the room")
    return parser

def parse_config():
    parser = create_parser()
    args = parser.parse_args()
    
    return AgentConfig(
        url=args.url,
        apikey=args.apikey,
        speed=args.speed,
        emotion=args.emotion.split(',') if args.emotion else [],
        prompt=args.prompt,
        voice_id=args.voice_id,
        session_time=args.session_time,
        token=args.token
    )

async def configure(aiohttp_session: aiohttp.ClientSession, config: Optional[AgentConfig] = None):
    if config is None:
        config = parse_config()

    if config.token:
        return (config.url, config.token)

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=config.apikey,
        daily_api_url="https://api.daily.co/v1",
        aiohttp_session=aiohttp_session,
    )

    token = await daily_rest_helper.get_token(config.url, config.session_time)
    return (config.url, token, config)
