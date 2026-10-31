"""
VaniShield AI — 4-Second Microphone Voice Recorder
Team CypherEye | Project VaniShield AI
Records high-fidelity 16kHz mono audio from the system microphone for live testing and enrollment.
"""

import os
import sys
import time
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "user_voice_sample.wav")

def record_microphone(duration=4.0, sample_rate=16000, output_path=OUTPUT_PATH):
    if not SOUNDDEVICE_AVAILABLE:
        print("[!] sounddevice is not installed. Install via: pip install sounddevice")
        return None

    print("\n" + "=" * 65)
    print(f" RECORDING MICROPHONE: {duration} SECONDS (16,000 Hz Mono)")
    print(" Speak naturally into your microphone...")
    print("=" * 65)

    try:
        devices = sd.query_devices()
        default_in = sd.query_devices(kind="input")
        print(f"[*] Input Device: {default_in['name']}")
    except Exception as e:
        print(f"[!] Warning querying audio devices: {e}")

    countdown = 3
    for c in range(countdown, 0, -1):
        print(f"  Starting in {c}...", end="\r", flush=True)
        time.sleep(1.0)
    print("  >>> RECORDING NOW! Speak clearly...        ", flush=True)

    try:
        frames = int(duration * sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("  [OK] Recording complete!")

        # Normalize audio signal
        audio_data = recording.flatten()
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-4:
            audio_data = audio_data / max_val * 0.90
        else:
            print("[!] Warning: Recorded audio is silent or near zero volume. Check microphone permissions.")

        sf.write(output_path, audio_data, sample_rate)
        print(f"[SUCCESS] Saved recorded sample -> {output_path} ({duration}s)\n")
        return output_path

    except Exception as e:
        print(f"[ERROR] Failed to record from microphone: {e}")
        return None

if __name__ == "__main__":
    record_microphone()
