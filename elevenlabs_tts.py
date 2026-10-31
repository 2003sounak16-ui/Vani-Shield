"""
ElevenLabs Text-to-Speech (TTS) Client for VaniShield AI
Supports neural voice synthesis for English, Hindi, Bengali, and multilingual clone simulations
Powered by ElevenLabs API (https://api.elevenlabs.io/v1/text-to-speech)
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any, List
import requests

logger = logging.getLogger("elevenlabs_tts")

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"

# Curated high-fidelity voices
CURATED_VOICES = {
    "rachel": {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel (Calm & Professional)", "gender": "female"},
    "adam": {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam (Deep & Authoritative)", "gender": "male"},
    "sarah": {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah (News & Formal)", "gender": "female"},
    "george": {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George (Warm & Persuasive)", "gender": "male"},
    "aria": {"id": "9BWtsMINqrJLrRacOk9x", "name": "Aria (Expressive & Dynamic)", "gender": "female"},
}
DEFAULT_VOICE_ID = CURATED_VOICES["rachel"]["id"]

def get_elevenlabs_api_key() -> str:
    """Retrieves ElevenLabs API key from environment variable."""
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()

def is_elevenlabs_available() -> bool:
    """Checks whether the ElevenLabs API key is configured."""
    return bool(get_elevenlabs_api_key())

def get_curated_voices() -> List[Dict[str, Any]]:
    """Returns the list of curated default ElevenLabs voices."""
    return [
        {"id": v["id"], "name": v["name"], "gender": v["gender"]}
        for v in CURATED_VOICES.values()
    ]

def fetch_account_voices(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches custom and cloned voices associated with the user's ElevenLabs account."""
    key = api_key or get_elevenlabs_api_key()
    if not key:
        return get_curated_voices()

    try:
        resp = requests.get(
            f"{ELEVENLABS_API_BASE}/voices",
            headers={"xi-api-key": key},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json().get("voices", [])
            return [
                {
                    "id": v.get("voice_id"),
                    "name": v.get("name"),
                    "category": v.get("category", "premade"),
                    "preview_url": v.get("preview_url")
                }
                for v in data
            ]
        else:
            logger.warning(f"Failed to fetch ElevenLabs voices (HTTP {resp.status_code}): {resp.text}")
            return get_curated_voices()
    except Exception as e:
        logger.warning(f"Error querying ElevenLabs voices: {e}")
        return get_curated_voices()

def synthesize_speech_elevenlabs(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_ELEVENLABS_MODEL,
    api_key: Optional[str] = None,
    output_format: str = "mp3_44100_128",
    timeout: int = 30
) -> Tuple[bytes, Dict[str, Any], Optional[str]]:
    """
    Synthesizes speech using ElevenLabs Text-to-Speech API.
    
    Args:
        text: The text to synthesize into spoken audio.
        voice_id: The ElevenLabs voice ID (default: Rachel).
        model_id: Model ID (default: "eleven_multilingual_v2").
        api_key: Optional ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.
        output_format: Desired audio format (default: "mp3_44100_128").
        timeout: Request timeout in seconds.
        
    Returns:
        tuple: (audio_bytes: bytes, metadata: dict, error_message: Optional[str])
    """
    key = api_key or get_elevenlabs_api_key()
    if not key:
        return b"", {}, "ELEVENLABS_API_KEY is not configured in environment or .env"

    if not text or not text.strip():
        return b"", {}, "Text input for synthesis cannot be empty."

    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    params = {"output_format": output_format}
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text.strip(),
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.80,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            params=params,
            headers=headers,
            timeout=timeout
        )

        if resp.status_code == 200:
            audio_bytes = resp.content
            metadata = {
                "voice_id": voice_id,
                "model_id": model_id,
                "audio_bytes_length": len(audio_bytes),
                "format": output_format,
                "engine": "ElevenLabs Neural TTS"
            }
            return audio_bytes, metadata, None

        elif resp.status_code == 401:
            err = "ElevenLabs 401 Unauthorized: Invalid or expired ELEVENLABS_API_KEY."
            logger.error(err)
            return b"", {}, err

        elif resp.status_code == 429:
            err = "ElevenLabs 429: Rate limit or character quota exceeded."
            logger.warning(err)
            return b"", {}, err

        else:
            err = f"ElevenLabs TTS API error (HTTP {resp.status_code}): {resp.text}"
            logger.error(err)
            return b"", {}, err

    except requests.exceptions.Timeout:
        err = f"ElevenLabs TTS API timed out after {timeout} seconds."
        logger.warning(err)
        return b"", {}, err
    except requests.exceptions.RequestException as e:
        err = f"ElevenLabs TTS network exception: {e}"
        logger.error(err)
        return b"", {}, err

def synthesize_with_fallback(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    api_key: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any], Optional[str]]:
    """
    Synthesizes speech using ElevenLabs if key is configured,
    or falls back gracefully to Edge-TTS if key is not yet provided.
    """
    key = api_key or get_elevenlabs_api_key()
    if key:
        return synthesize_speech_elevenlabs(text=text, voice_id=voice_id, api_key=key)

    # Fallback to Edge-TTS
    try:
        import asyncio
        import io
        import edge_tts

        # Pick suitable voice for text language
        voice = "en-IN-PrabhatNeural"
        if any('\u0980' <= c <= '\u09FF' for c in text):
            voice = "bn-IN-BashkarNeural"
        elif any('\u0900' <= c <= '\u097F' for c in text):
            voice = "hi-IN-MadhurNeural"

        async def _edge_gen():
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                audio_bytes = pool.submit(asyncio.run, _edge_gen()).result()
        else:
            audio_bytes = asyncio.run(_edge_gen())

        metadata = {
            "voice_id": voice,
            "engine": "Edge-TTS (Fallback - Set ELEVENLABS_API_KEY to unlock ElevenLabs)"
        }
        return audio_bytes, metadata, None

    except Exception as e:
        return b"", {}, f"Synthesis failed (both ElevenLabs key unset and Edge-TTS failed: {e})"
