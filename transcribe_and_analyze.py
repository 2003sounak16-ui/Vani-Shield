import os
import sys
import glob
import json
import librosa
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faster_whisper import WhisperModel

def run():
    asr = WhisperModel("base", device="cpu", compute_type="int8")
    files = sorted(glob.glob(r"c:\AI\VoiceCloning\audio\*"))

    results = []
    for f in files:
        fname = os.path.basename(f)
        y, sr = librosa.load(f, sr=16000)
        dur = len(y) / 16000.0
        
        segments, info = asr.transcribe(y, beam_size=5)
        seg_list = []
        for s in segments:
            seg_list.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip()
            })
        full_text = " ".join([s["text"] for s in seg_list])

        item = {
            "file": fname,
            "duration": round(dur, 2),
            "language": info.language,
            "lang_prob": round(info.language_probability, 3),
            "full_text": full_text,
            "segments": seg_list
        }
        results.append(item)
        print(f"=== {fname} ===")
        print(f"Language: {info.language} ({info.language_probability:.2f}) | Duration: {dur:.2f}s")
        print(f"Transcript: {full_text}")
        print()

    with open(r"c:\AI\VoiceCloning\audio_transcripts.json", "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run()
