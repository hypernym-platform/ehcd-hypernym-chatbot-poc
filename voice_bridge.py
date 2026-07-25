"""
Bridge between our app and ElevenLabs' real-time Conversational AI (voice-to-
voice) product.

We never relay raw audio ourselves — the client connects directly to
ElevenLabs' own WebSocket for that. This module only handles the two
server-side pieces:
  1. Minting the signed URL the client uses to open that connection
     (keeps our raw ElevenLabs API key off the client).
  2. Minting/verifying a short-lived internal token that identifies which
     of our app's users is on a given voice call, since ElevenLabs' Custom
     LLM integration authenticates with one static shared secret for all
     calls, not a per-user credential. The client passes this token back to
     ElevenLabs as a `voice_session_token` dynamic variable, the agent's
     system prompt embeds it as literal text, and our /v1/chat/completions
     bridge (in app.py) parses it back out.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import jwt

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
VOICE_SESSION_TOKEN_TTL_SECONDS = 3600  # covers a generously long call
_JWT_ALGORITHM = "HS256"


class VoiceBridgeError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


async def get_elevenlabs_signed_url(agent_id: str, api_key: str) -> str:
    """Ask ElevenLabs for a signed WebSocket URL for this agent. Valid 15
    minutes to *open* the connection (per ElevenLabs); the conversation
    itself can run longer once started."""
    if not api_key:
        raise VoiceBridgeError(500, "ElevenLabs API key is not configured")
    if not agent_id:
        raise VoiceBridgeError(500, "ElevenLabs agent id is not configured")

    url = f"{ELEVENLABS_BASE_URL}/convai/conversation/get-signed-url"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                url,
                headers={"xi-api-key": api_key},
                params={"agent_id": agent_id},
            )
    except httpx.HTTPError as e:
        logger.error(f"[ElevenLabs] get_signed_url request failed: {e}")
        raise VoiceBridgeError(502, f"Could not reach ElevenLabs: {e}")

    if resp.status_code != 200:
        detail = resp.text[:500]
        logger.error(f"[ElevenLabs] get_signed_url {resp.status_code}: {detail}")
        raise VoiceBridgeError(resp.status_code, detail or "get_signed_url failed")

    data = resp.json()
    signed_url = data.get("signed_url")
    if not signed_url:
        raise VoiceBridgeError(502, "ElevenLabs response missing signed_url")
    return signed_url


def mint_voice_session_token(user_id: int, secret: str, expires_in_seconds: int) -> str:
    """A short-lived, purpose-scoped token — NOT the same as the app's normal
    login JWT — that only proves 'this user_id started a voice session
    recently', for RBAC identification inside the Custom LLM bridge."""
    now = datetime.now(timezone.utc)
    payload = {
        "purpose": "voice_session",
        "user_id": user_id,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in_seconds),
    }
    return jwt.encode(payload, secret, algorithm=_JWT_ALGORITHM)


def verify_voice_session_token(token: str, secret: str) -> Optional[int]:
    """Returns user_id if the token is valid, unexpired, and voice-scoped; None otherwise."""
    try:
        payload = jwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError as e:
        logger.warning(f"[VoiceBridge] Invalid voice session token: {e}")
        return None

    if payload.get("purpose") != "voice_session":
        logger.warning("[VoiceBridge] Token presented is not a voice_session token")
        return None

    user_id = payload.get("user_id")
    return int(user_id) if user_id is not None else None
