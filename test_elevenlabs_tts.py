"""
Quick test tool for ElevenLabs Text-to-Speech (TTS)
Usage:
  python test_elevenlabs_tts.py ["Your test scam or normal message"] [optional_voice_id]
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

from elevenlabs_tts import synthesize_with_fallback, is_elevenlabs_available, CURATED_VOICES

def main():
    print("=" * 65)
    print("ElevenLabs Text-to-Speech (TTS) Synthesis Verification")
    print("=" * 65)

    if not is_elevenlabs_available():
        print("ℹ️  ELEVENLABS_API_KEY is not set in .env. Falling back to Edge-TTS.")
        print("   To use ElevenLabs neural models, set ELEVENLABS_API_KEY=your_key in .env")
        print("   Sign up / obtain key at: https://elevenlabs.io/\n")

    text = (
        sys.argv[1] if len(sys.argv) > 1
        else "Urgent security notice from State Bank of India fraud team. Please verify your OTP immediately."
    )
    voice_id = sys.argv[2] if len(sys.argv) > 2 else CURATED_VOICES["rachel"]["id"]
    output_path = "elevenlabs_test_output.mp3"

    print(f"Synthesizing text: \"{text}\"")
    print(f"Target Voice ID: {voice_id}")

    audio_bytes, meta, err = synthesize_with_fallback(text, voice_id=voice_id)

    if err and not audio_bytes:
        print(f"❌ Synthesis failed: {err}")
    else:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Generated {len(audio_bytes)} audio bytes using: {meta.get('engine')}")
        print(f"Saved audio output to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
