"""
Interactive real-time voice-to-voice test — NOT part of the app.
Speak into your mic; hear the agent's real spoken reply through your speakers.
Ctrl+C to stop.
"""

import asyncio
import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx
import jwt
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
BACKEND_URL = "http://localhost:8080"
SAMPLE_RATE = 16000
CHUNK_MS = 250
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


def mint_app_token(user_id: int = 1) -> str:
    payload = {"user_id": user_id, "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def out(line: str):
    # Force UTF-8 regardless of the Windows console's default codepage.
    sys.stdout.buffer.write((line + "\n").encode("utf-8"))
    sys.stdout.flush()


async def main():
    app_token = mint_app_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BACKEND_URL}/api/voice/signed-url",
            headers={"Authorization": f"Bearer {app_token}"},
            timeout=15.0,
        )
    resp.raise_for_status()
    data = resp.json()
    signed_url = data["signed_url"]
    dynamic_variables = data["dynamic_variables"]
    out("[+] Got signed_url from our backend. Connecting to ElevenLabs...")

    loop = asyncio.get_event_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()

    out_stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    out_stream.start()

    def mic_callback(indata, frames, time_info, status):
        if status:
            out(f"  [mic status: {status}]")
        loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

    in_stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=mic_callback,
    )

    async with websockets.connect(signed_url) as ws:
        await ws.send(json.dumps({
            "type": "conversation_initiation_client_data",
            "dynamic_variables": dynamic_variables,
        }))
        out("[+] Connected. Speak now! (Ctrl+C to stop)\n")

        async def send_mic_audio():
            in_stream.start()
            try:
                while True:
                    chunk = await mic_queue.get()
                    b64 = base64.b64encode(chunk).decode("ascii")
                    await ws.send(json.dumps({"user_audio_chunk": b64}))
            finally:
                in_stream.stop()
                in_stream.close()

        async def receive_and_play():
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "audio":
                    b64 = msg.get("audio_event", {}).get("audio_base_64", "")
                    if b64:
                        out_stream.write(base64.b64decode(b64))
                elif msg_type == "agent_response":
                    out(f"  Agent: {msg['agent_response_event']['agent_response']}")
                elif msg_type == "user_transcript":
                    text = msg.get("user_transcription_event", {}).get("user_transcript", "")
                    if text:
                        out(f"  You said: {text}")
                elif msg_type == "interruption":
                    out("  [agent interrupted itself]")

        try:
            await asyncio.gather(send_mic_audio(), receive_and_play())
        finally:
            out_stream.stop()
            out_stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[i] Stopped.")
