"""
Sarvam AI Speech-to-Text (STT) Client for VaniShield AI
Supports high-accuracy Indic language & Indian English speech recognition
Powered by Sarvam AI Saaras model (https://api.sarvam.ai/speech-to-text)
"""

import os
import io
import logging
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger("sarvam_stt")

SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_MODEL = "saaras:v2"

def get_sarvam_api_key() -> str:
    """Retrieves Sarvam API key from environment variable."""
    return os.environ.get("SARVAM_API_KEY", "").strip()

def is_sarvam_available() -> bool:
    """Checks whether the Sarvam API key is configured."""
    return bool(get_sarvam_api_key())

def audio_to_wav_bytes(
    audio_data: Union[np.ndarray, bytes, str],
    sample_rate: int = 16000
) -> bytes:
    """Converts numpy array, file path, or raw audio bytes into 16kHz PCM WAV bytes."""
    if isinstance(audio_data, str):
        if not os.path.exists(audio_data):
            raise FileNotFoundError(f"Audio file not found: {audio_data}")
        with open(audio_data, "rb") as f:
            return f.read()

    elif isinstance(audio_data, np.ndarray):
        # Ensure float32 or int16 range
        arr = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        buf = io.BytesIO()
        sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    elif isinstance(audio_data, (bytes, bytearray)):
        return bytes(audio_data)

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_data)}")

def transcribe_with_sarvam(
    audio_input: Union[np.ndarray, bytes, str],
    sample_rate: int = 16000,
    api_key: Optional[str] = None,
    model: str = DEFAULT_SARVAM_MODEL,
    language_code: str = "unknown",
    timeout: int = 5
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Transcribes audio using Sarvam AI's Saaras speech-to-text API.
    
    Args:
        audio_input: Audio as 1D numpy array, file path, or audio bytes.
        sample_rate: Audio sampling rate (default: 16000).
        api_key: Optional Sarvam subscription key. Falls back to SARVAM_API_KEY env var.
        model: Model identifier (default: "saaras:v2", also supports "saaras:v1").
        language_code: BCP-47 language tag (e.g. "hi-IN", "bn-IN", "en-IN", or "unknown" for auto-detection).
        timeout: Request timeout in seconds.
        
    Returns:
        tuple: (transcript: str, metadata: dict, error_message: Optional[str])
    """
    key = api_key or get_sarvam_api_key()
    if not key:
        return "", {}, "SARVAM_API_KEY is not configured in environment or .env"

    try:
        wav_bytes = audio_to_wav_bytes(audio_input, sample_rate=sample_rate)
    except Exception as e:
        return "", {}, f"Failed to encode audio for Sarvam STT: {e}"

    headers = {
        "api-subscription-key": key
    }
    files = {
        "file": ("input_audio.wav", wav_bytes, "audio/wav")
    }
    data = {
        "model": model,
        "language_code": language_code,
        "mode": "transcribe"
    }

    try:
        resp = requests.post(
            SARVAM_STT_ENDPOINT,
            files=files,
            data=data,
            headers=headers,
            timeout=timeout
        )

        if resp.status_code == 200:
            res_json = resp.json()
            transcript = res_json.get("transcript", "").strip()
            metadata = {
                "request_id": res_json.get("request_id"),
                "language_code": res_json.get("language_code", language_code),
                "model": model,
                "engine": "Sarvam AI Saaras"
            }
            return transcript, metadata, None

        elif resp.status_code == 401:
            err = "Sarvam AI 401 Unauthorized: Invalid or expired SARVAM_API_KEY."
            logger.error(err)
            return "", {}, err

        elif resp.status_code == 429:
            err = "Sarvam AI 429 Rate Limit Exceeded: Please check your Sarvam quota."
            logger.warning(err)
            return "", {}, err

        else:
            err = f"Sarvam AI STT API error (HTTP {resp.status_code}): {resp.text}"
            logger.error(err)
            return "", {}, err

    except requests.exceptions.Timeout:
        err = f"Sarvam AI STT API timed out after {timeout} seconds."
        logger.warning(err)
        return "", {}, err
    except requests.exceptions.RequestException as e:
        err = f"Sarvam AI STT network exception: {e}"
        logger.error(err)
        return "", {}, err
