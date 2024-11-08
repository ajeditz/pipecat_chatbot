from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import aiohttp
import argparse
import os
import json
import base64
import subprocess
from dotenv import load_dotenv
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

load_dotenv(override=True)

class BotConfig(BaseModel):
    speed: str = Field("normal", description="Voice speed (slow/normal/fast)")
    emotion: List[str] = Field(["positivity:high", "curiosity"], description="List of emotions for the voice")
    prompt: str = Field("You are a friendly customer service agent...", description="System prompt for the bot")
    voice_id: str = Field("a0e99841-438c-4a64-b679-ae501e7d6091", description="Voice ID for TTS")
    session_time: Optional[float] = Field(3600, description="Session expiry time in seconds")

MAX_BOTS_PER_ROOM = 1
bot_procs = {}
daily_helpers = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/start_bot")
async def start_agent(config: BotConfig):
    try:
        # Create room
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")

        # Check bot limit
        num_bots_in_room = sum(
            1 for proc in bot_procs.values() 
            if proc[1] == room.url and proc[0].poll() is None
        )
        if num_bots_in_room >= MAX_BOTS_PER_ROOM:
            raise HTTPException(
                status_code=500,
                detail=f"Max bot limit reached for room: {room.url}"
            )

        # Get token
        token = await daily_helpers["rest"].get_token(
            room.url, 
            expiry_time=config.session_time
        )

        if not token:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get token for room: {room.url}"
            )

        # Convert config to base64 to safely pass it as a command line argument
        config_str = json.dumps(config.dict())
        config_b64 = base64.b64encode(config_str.encode()).decode()

        # Start bot process with configuration
        cmd = f"python3 bot2.py --url {room.url} --token {token} --config {config_b64}"
        proc = subprocess.Popen(
            cmd,
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room.url)

        return {
            "room_url": room.url,
            "token": token,
            "bot_pid": proc.pid,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/status/{pid}")
def get_status(pid: int):
    proc = bot_procs.get(pid)
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")
    
    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})

if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "8080"))

    parser = argparse.ArgumentParser(description="Daily Voice Agent FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
