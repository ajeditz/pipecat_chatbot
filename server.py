import aiohttp
import os
import argparse
import subprocess
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import json
import base64

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams, DailyRoomProperties

from dotenv import load_dotenv

load_dotenv(override=True)


class BotConfig(BaseModel):
    speed: str = Field("normal", description="Voice speed (slow/normal/fast)")
    emotion: list[str] = Field(["positivity:high", "curiosity"], description="List of emotions for the voice")
    prompt: str = Field("You are a friendly customer service agent...", description="System prompt for the bot")
    voice_id: str = Field("voice_id_here", description="Voice ID for TTS")
    session_time: float = Field(description="Session expiry time in min.")


MAX_BOTS_PER_ROOM = 1

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}

daily_helpers = {}


def cleanup():
    # Clean up function, just to be extra safe
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


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
    cleanup()


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
    print(f"!!! Creating room")
    room = await daily_helpers["rest"].create_room(DailyRoomParams(properties=DailyRoomProperties(exp= time.time() + (config.session_time) * 60)))
    print(f"!!! Room URL: {room.url}")
    # Ensure the room property is present
    if not room.url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!",
        )

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room.url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limited reach for room: {room.url}")

    # Get the token for the room
    token = await daily_helpers["rest"].get_token(room.url)

    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")


    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)


    try:
        config_str = json.dumps(config.model_dump())
        config_b64 = base64.b64encode(config_str.encode()).decode()

        cmd = f"python3 bot.py --url {room.url} --token {token} --config {config_b64}"
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
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc[0].poll() is None:
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status})


if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    configuration = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=configuration.host,
        port=configuration.port,
        reload=configuration.reload,
    )
