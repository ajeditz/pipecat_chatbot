import asyncio
import aiohttp
import json
import base64
import os
import sys
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import (
    OutputImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TranscriptionFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Load sprite animations
sprites = []
script_dir = os.path.dirname(__file__)
for i in range(1, 26):
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)

class TalkingAnimation(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame)

class TranscriptionCollector(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.transcripts = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self.transcripts.append(frame.transcription)

        await self.push_frame(frame)

async def main(url, token, config_b64):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            url,
            token,
            "Voice Agent",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            ),
        )

        # Decode and parse configuration
        config_str = base64.b64decode(config_b64).decode()
        config = json.loads(config_str)

        # Configure TTS with provided parameters
        tts_params = CartesiaTTSService.InputParams(
            speed=config["speed"],
            emotion=config["emotion"]
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=config["voice_id"],
            params=tts_params
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4"
        )

        # Initialize with provided prompt
        messages = [
            {
                "role": "system",
                "content": config["prompt"]
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        ta = TalkingAnimation()
        tc = TranscriptionCollector()

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                ta,
                tc,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
        await task.queue_frame(quiet_frame)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()
        await runner.run(task)
        return tc.transcripts

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Agent Bot")
    parser.add_argument("--url", required=True, help="Daily room URL")
    parser.add_argument("--token", required=True, help="Daily room token")
    parser.add_argument("--config", required=True, help="Base64 encoded configuration")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.token, args.config))