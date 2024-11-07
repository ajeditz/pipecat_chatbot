import aiohttp
import os
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

async def configure(aiohttp_session: aiohttp.ClientSession, room_url: str, expiry_time: float):
    """
    Configure the Daily room with the provided parameters
    
    Args:
        aiohttp_session: Active aiohttp client session
        room_url: URL of the Daily room
        expiry_time: Token expiry time in seconds
    
    Returns:
        tuple: (room_url, token)
    """
    key = os.getenv("DAILY_API_KEY")
    
    if not key:
        raise Exception(
            "No Daily API key specified. Set DAILY_API_KEY in your environment."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    token = await daily_rest_helper.get_token(room_url, expiry_time)
    return (room_url, token)
