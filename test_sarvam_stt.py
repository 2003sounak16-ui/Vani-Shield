"""
Quick test tool for Sarvam AI Speech-to-Text (STT) Saaras model
Usage:
  python test_sarvam_stt.py [optional_path_to_audio_file]
"""

import os
import sys
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

from sarvam_stt import transcribe_with_sarvam, is_sarvam_available

def main():
    print("=" * 65)
    print("Sarvam AI Speech-to-Text (STT) Saaras Verification")
    print("=" * 65)

    if not is_sarvam_available():
        print("⚠️  SARVAM_API_KEY is not set in your .env file or environment.")
        print("   Please set SARVAM_API_KEY=your_key in C:\\AI\\VoiceCloning\\.env")
        print("   Sign up / obtain key at: https://www.sarvam.ai/")
        print("   (VaniShield will use Faster-Whisper as local fallback in the meantime.)")
        return

    # Choose audio file to test
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        audio_file = sys.argv[1]
    else:
        candidates = (
            glob.glob("test_multilingual_calls/*.wav") +
            glob.glob("audio/*.wav") +
            glob.glob("data_demo/*/*.wav")
        )
        if not candidates:
            print("No test audio files found in repository to test.")
            return
        audio_file = candidates[0]

    print(f"Testing Sarvam AI STT on: {audio_file}")
    transcript, meta, err = transcribe_with_sarvam(audio_file)

    if err:
        print(f"❌ Transcription Failed: {err}")
    else:
        print(f"✅ Success!")
        print(f"Detected Language: {meta.get('language_code', 'unknown')}")
        print(f"Model: {meta.get('model')}")
        print(f"Transcript: {transcript}")

if __name__ == "__main__":
    main()
