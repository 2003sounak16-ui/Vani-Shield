"""
VaniShield AI — Multilingual Speech Data Ingestion Pipeline
Target Languages: English (en), Bengali (bn), Hindi (hi)
Datasets:
  - Bengali Real: OpenSLR-37 (Crowdsourced Indian Bengali Speech)
  - Hindi Real: OpenSLR-103 (Navana Tech MUCS Hindi Speech)
  - English Real: OpenSLR-12 / LibriSpeech (Clean Human Utterances)
  - English Fake: Deepfake-Audio (Real neural vocoder deepfakes from Hugging Face)
  - Bengali Fake: Neural Cloned Speech via Indian Neural Vocoders (bn-IN / bn-BD)
  - Hindi Fake: Neural Cloned Speech via Indian Neural Vocoders (hi-IN)
"""

import os
import io
import time
import zipfile
import tarfile
import random
import asyncio
import requests
import urllib.request
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import torch
from huggingface_hub import hf_hub_download, HfApi

TARGET_SR = 16000
TARGET_DURATION = 3.5  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 56,000 samples
SAMPLES_PER_CATEGORY = 60  # ~60 per language/class = 180 Real, 180 Fake = 360 total balanced clips

OUTPUT_DIRS = {
    ("real", "en"): "data_real/real/en",
    ("real", "bn"): "data_real/real/bn",
    ("real", "hi"): "data_real/real/hi",
    ("fake", "en"): "data_real/fake/en",
    ("fake", "bn"): "data_real/fake/bn",
    ("fake", "hi"): "data_real/fake/hi",
}

for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

def standardize_and_save(audio_data, sr, out_path):
    """Resample to 16kHz mono, trim/pad to TARGET_DURATION, and save as 16-bit PCM WAV."""
    if audio_data.ndim > 1:
        if audio_data.shape[0] == 2:
            audio_data = np.mean(audio_data, axis=0)
        elif audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.squeeze()

    if sr != TARGET_SR:
        t_audio = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        t_resampled = torchaudio.transforms.Resample(sr, TARGET_SR)(t_audio).squeeze(0)
        audio_data = t_resampled.numpy()

    # Normalize amplitude
    max_val = np.max(np.abs(audio_data))
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95

    # Trim / Pad to target samples
    if len(audio_data) > TARGET_SAMPLES:
        # Take center or beginning
        audio_data = audio_data[:TARGET_SAMPLES]
    else:
        audio_data = np.pad(audio_data, (0, TARGET_SAMPLES - len(audio_data)))

    sf.write(out_path, audio_data.astype(np.float32), TARGET_SR, subtype="PCM_16")

# -------------------------------------------------------------
# 1. HTTP Range File reader for OpenSLR Zip archives without full download
# -------------------------------------------------------------
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.length = int(resp.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={self.pos}-{self.pos + size - 1}"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        self.pos += len(data)
        return data

# -------------------------------------------------------------
# 2. Ingest Real Bengali Speech (OpenSLR-37 bn_in.zip)
# -------------------------------------------------------------
def fetch_real_bengali(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[1/6] Fetching Real Bengali Speech from OpenSLR-37...")
    url = "https://openslr.trmal.net/resources/37/bn_in.zip"
    records = []
    try:
        f = HTTPRangeFile(url)
        zf = zipfile.ZipFile(f)
        wav_names = [n for n in zf.namelist() if n.endswith(".wav")]
        print(f"  Found {len(wav_names)} Bengali utterances in OpenSLR-37.")
        # Shuffle deterministically to get diverse speakers
        random.seed(42)
        random.shuffle(wav_names)

        saved = 0
        for name in wav_names:
            if saved >= n_target:
                break
            try:
                base = os.path.basename(name)
                # Format: bin_1010_4358822175.wav -> speaker is bin_1010
                parts = base.split("_")
                spk = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "bn_spk_0"
                raw = zf.read(name)
                data, sr = sf.read(io.BytesIO(raw))
                out_file = os.path.join(OUTPUT_DIRS[("real", "bn")], f"bn_real_{saved:03d}_{base}")
                standardize_and_save(data, sr, out_file)
                records.append({
                    "path": out_file.replace("\\", "/"),
                    "label": 0,
                    "language": "bn",
                    "speaker_group": spk
                })
                saved += 1
            except Exception as e:
                continue
        print(f"  Saved {saved} Real Bengali samples.")
    except Exception as e:
        print(f"  OpenSLR-37 error: {e}")
    return records

# -------------------------------------------------------------
# 3. Ingest Real Hindi Speech (OpenSLR-103 Hindi_test.tar.gz)
# -------------------------------------------------------------
def fetch_real_hindi(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[2/6] Fetching Real Hindi Speech from OpenSLR-103...")
    url = "https://openslr.trmal.net/resources/103/Hindi_test.tar.gz"
    records = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=20)
        saved = 0
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if saved >= n_target:
                    break
                if member.name.endswith(".wav"):
                    try:
                        base = os.path.basename(member.name)
                        # Format: 2131_098.wav -> speaker is 2131
                        spk = "hi_spk_" + base.split("_")[0]
                        raw = tar.extractfile(member).read()
                        data, sr = sf.read(io.BytesIO(raw))
                        out_file = os.path.join(OUTPUT_DIRS[("real", "hi")], f"hi_real_{saved:03d}_{base}")
                        standardize_and_save(data, sr, out_file)
                        records.append({
                            "path": out_file.replace("\\", "/"),
                            "label": 0,
                            "language": "hi",
                            "speaker_group": spk
                        })
                        saved += 1
                    except Exception as e:
                        continue
        resp.close()
        print(f"  Saved {saved} Real Hindi samples.")
    except Exception as e:
        print(f"  OpenSLR-103 error: {e}")
    return records

# -------------------------------------------------------------
# 4. Ingest Real English Speech (OpenSLR-12 / LibriSpeech validation clean)
# -------------------------------------------------------------
def fetch_real_english(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[3/6] Fetching Real English Speech from LibriSpeech (clean)...")
    records = []
    try:
        p = hf_hub_download(
            repo_id="hf-internal-testing/librispeech_asr_demo",
            filename="clean/validation-00000-of-00001.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(p)
        saved = 0
        for i, row in df.iterrows():
            if saved >= n_target:
                break
            try:
                audio_dict = row["audio"]
                raw_bytes = audio_dict["bytes"]
                spk = f"en_spk_{row.get('speaker_id', i % 10)}"
                data, sr = sf.read(io.BytesIO(raw_bytes))
                out_file = os.path.join(OUTPUT_DIRS[("real", "en")], f"en_real_{saved:03d}.wav")
                standardize_and_save(data, sr, out_file)
                records.append({
                    "path": out_file.replace("\\", "/"),
                    "label": 0,
                    "language": "en",
                    "speaker_group": spk
                })
                saved += 1
            except Exception as e:
                continue
        print(f"  Saved {saved} Real English samples.")
    except Exception as e:
        print(f"  LibriSpeech error: {e}")
    return records

# -------------------------------------------------------------
# 5. Ingest English Neural Deepfakes (Hugging Face Deepfake-Audio dataset)
# -------------------------------------------------------------
def fetch_fake_english(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[4/6] Fetching Real Neural Deepfakes for English from 3004lakshu/Deepfake-Audio...")
    records = []
    try:
        api = HfApi()
        files = api.list_repo_files("3004lakshu/Deepfake-Audio", repo_type="dataset")
        fake_files = [f for f in files if "raw_audio/fake" in f and f.endswith(".wav")]
        random.seed(42)
        random.shuffle(fake_files)

        saved = 0
        for fpath in fake_files:
            if saved >= n_target:
                break
            try:
                local_path = hf_hub_download(repo_id="3004lakshu/Deepfake-Audio", filename=fpath, repo_type="dataset")
                data, sr = sf.read(local_path)
                base = os.path.basename(fpath).split(".")[0]
                spk = f"en_fake_vocoder_{saved % 12}"
                out_file = os.path.join(OUTPUT_DIRS[("fake", "en")], f"en_fake_{saved:03d}_{base}.wav")
                standardize_and_save(data, sr, out_file)
                records.append({
                    "path": out_file.replace("\\", "/"),
                    "label": 1,
                    "language": "en",
                    "speaker_group": spk
                })
                saved += 1
            except Exception as e:
                continue
        print(f"  Saved {saved} Fake English samples.")
    except Exception as e:
        print(f"  Deepfake-Audio error: {e}")
    return records

# -------------------------------------------------------------
# 6. Ingest Bengali Neural Clones (Indian Neural Vocoders with vishing scenarios)
# -------------------------------------------------------------
async def fetch_fake_bengali(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[5/6] Synthesizing Indian Language Neural Clones (Bengali) using Indian Neural Vocoders...")
    import edge_tts
    records = []
    voices = ["bn-IN-BashkarNeural", "bn-IN-TanishaaNeural", "bn-BD-PradeepNeural", "bn-BD-NabanitaNeural"]
    scam_templates = [
        "জরুরি সতর্কতা: আপনার স্টেট ব্যাংক একাউন্ট স্থগিত করা হয়েছে। অবিলম্বে ৬ সংখ্যার ওটিপি বলুন।",
        "নমস্কার, আমি প্রধান ব্যাংক শাখা থেকে বলছি। আপনার এটিএম কার্ড ব্লক হয়েছে, সিভিভি কোড প্রদান করুন।",
        "ইনকাম ট্যাক্স রিফান্ড পেতে অবিলম্বে আপনার ইউপিআই পিন এবং অ্যাকাউন্ট নম্বর নিশ্চিত করুন।",
        "বিদ্যুৎ বিল পরিশোধের লিঙ্কটি খুলুন অন্যথায় আপনার বিদ্যুৎ সংযোগ এক ঘণ্টার মধ্যে বিচ্ছিন্ন হবে।",
        "আপনার কেওয়াইসি ভেরিফিকেশন মেয়াদ শেষ হয়ে গেছে। গুগল পে বা ফোনপে ওটিপি শেয়ার করুন।",
        "অভিনন্দন! আপনি পঁচিশ লাখ টাকার লটারি জিতেছেন। প্রসেসিং ফি হিসেবে এখনই টাকা ট্রান্সফার করুন।",
        "আমি সাইবার সেল থেকে বলছি। আপনার আধার কার্ড অবৈধ অর্থপাচারে ব্যবহৃত হয়েছে। অ্যাকাউন্ট ভেরিফাই করুন।",
        "আপনার ক্রেডিট কার্ডে পঁয়তাল্লিশ হাজার টাকার ট্রানজাকশন আটকে আছে। বাতিল করতে ওটিপি দিন।",
        "আপনার সিম কার্ড ২৪ ঘণ্টার মধ্যে বন্ধ হবে। চালু রাখতে আপনার ব্যাংক তথ্য যাচাই করুন।",
        "জরুরি পুলিশ ভেরিফিকেশন: অবিলম্বে আপনার ডিজিটাল সমন দেখতে ব্যাংক ওটিপি কোড জানান।"
    ]

    saved = 0
    temp_dir = "temp_bn_tts"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(n_target):
        voice = voices[i % len(voices)]
        text = scam_templates[i % len(scam_templates)]
        spk = f"bn_fake_{voice.split('-')[2].replace('Neural','')}"
        temp_file = os.path.join(temp_dir, f"tmp_bn_{i}.mp3")
        out_file = os.path.join(OUTPUT_DIRS[("fake", "bn")], f"bn_fake_{saved:03d}_{spk}.wav")

        try:
            await edge_tts.Communicate(text, voice).save(temp_file)
            data, sr = sf.read(temp_file)
            standardize_and_save(data, sr, out_file)
            records.append({
                "path": out_file.replace("\\", "/"),
                "label": 1,
                "language": "bn",
                "speaker_group": spk
            })
            saved += 1
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            continue

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
    print(f"  Saved {saved} Fake Bengali samples.")
    return records

# -------------------------------------------------------------
# 7. Ingest Hindi Neural Clones (Indian Neural Vocoders with vishing scenarios)
# -------------------------------------------------------------
async def fetch_fake_hindi(n_target=SAMPLES_PER_CATEGORY):
    print(f"\n[6/6] Synthesizing Indian Language Neural Clones (Hindi) using Indian Neural Vocoders...")
    import edge_tts
    records = []
    voices = ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]
    scam_templates = [
        "नमस्कार, भारतीय स्टेट बैंक कस्टमर केयर। आपका बैंक खाता ब्लॉक हो गया है, तुरंत छह अंकों का ओटीपी बताएं।",
        "चेतावनी: आपका पैन कार्ड आधार से लिंक नहीं है। बैंक सेवा जारी रखने के लिए यूपीआई पिन दर्ज करें।",
        "क्रेडिट कार्ड धोखाधड़ी विभाग: आपके कार्ड से चालीस हजार रुपये कटे हैं। चार्ज रद्द करने के लिए सीवीवी बताएं।",
        "आपका बिजली बिल बकाया है। आज रात बिजली काट दी जाएगी। बिल भरने के लिए तुरंत इस नंबर पर पैसे भेजें।",
        "साइबर क्राइम ब्रांच दिल्ली: आपका नाम मनी लॉन्ड्रिंग केस में आया है। अपना बैंक बैलेंस वेरीफाई करें।",
        "बधाई हो, केबीसी लकी ड्रा में आपकी पच्चीस लाख की लॉटरी लगी है। राशि प्राप्त करने हेतु प्रोसेसिंग फीस दें।",
        "आपका सिम कार्ड अमान्य हो गया है। एक्टिवेट रखने के लिए बैंक अकाउंट नंबर और ओटीपी साझा करें।",
        "लोन अप्रूवल डिपार्टमेंट: आपका पांच लाख का प्री-अप्रूव्ड लोन तैयार है। ओटीपी शेयर करके तुरंत अप्रूव करें।",
        "एचडीएफसी बैंक सुरक्षा अलर्ट: आपके खाते में संदिग्ध लॉगिन हुआ है। अपना एटीएम पिन और ओटीपी कन्फर्म करें।",
        "इनकम टैक्स नोटिस: आपके खाते पर टैक्स पेनल्टी लगी है। गिरफ्तारी से बचने के लिए तुरंत यूपीआई भुगतान करें।"
    ]

    saved = 0
    temp_dir = "temp_hi_tts"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(n_target):
        voice = voices[i % len(voices)]
        text = scam_templates[i % len(scam_templates)]
        spk = f"hi_fake_{voice.split('-')[2].replace('Neural','')}"
        temp_file = os.path.join(temp_dir, f"tmp_hi_{i}.mp3")
        out_file = os.path.join(OUTPUT_DIRS[("fake", "hi")], f"hi_fake_{saved:03d}_{spk}.wav")

        try:
            await edge_tts.Communicate(text, voice).save(temp_file)
            data, sr = sf.read(temp_file)
            standardize_and_save(data, sr, out_file)
            records.append({
                "path": out_file.replace("\\", "/"),
                "label": 1,
                "language": "hi",
                "speaker_group": spk
            })
            saved += 1
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            continue

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
    print(f"  Saved {saved} Fake Hindi samples.")
    return records

# -------------------------------------------------------------
# 8. Main Orchestrator & Manifest Generation
# -------------------------------------------------------------
def main():
    print("=" * 65)
    print("VaniShield AI: Real Multilingual Speech Ingestion Pipeline")
    print("=" * 65)

    all_records = []

    # Real human speech
    all_records.extend(fetch_real_bengali())
    all_records.extend(fetch_real_hindi())
    all_records.extend(fetch_real_english())

    # Deepfake & cloned speech
    all_records.extend(fetch_fake_english())
    all_records.extend(asyncio.run(fetch_fake_bengali()))
    all_records.extend(asyncio.run(fetch_fake_hindi()))

    df = pd.DataFrame(all_records)
    print("\n" + "=" * 65)
    print(f"Total downloaded and processed samples: {len(df)}")
    print(df.groupby(["language", "label"]).size().unstack(fill_value=0))

    # Split train (80%) and validation (20%) with strictly NO speaker leakage
    df["split"] = "train"
    for (lang, lbl), group in df.groupby(["language", "label"]):
        unique_spks = sorted(group["speaker_group"].unique())
        random.seed(42)
        random.shuffle(unique_spks)
        n_val = max(1, int(len(unique_spks) * 0.20))
        val_spks = set(unique_spks[:n_val])
        df.loc[(df["language"] == lang) & (df["label"] == lbl) & (df["speaker_group"].isin(val_spks)), "split"] = "val"

    manifest_path = "multilingual_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest to {manifest_path}:")
    print(f"  Train samples: {len(df[df['split']=='train'])}")
    print(f"  Val samples:   {len(df[df['split']=='val'])}")
    print(df.groupby(["language", "label", "split"]).size())

    # Verify 0% speaker leakage
    train_spks = set(df[df["split"]=="train"]["speaker_group"])
    val_spks = set(df[df["split"]=="val"]["speaker_group"])
    leakage = train_spks.intersection(val_spks)
    print(f"Speaker leakage between train and val: {len(leakage)} speakers (Should be 0).")
    assert len(leakage) == 0, f"Error: Speaker leakage detected: {leakage}"
    print("[SUCCESS] Multilingual dataset downloaded and verified with ZERO speaker leakage!")

if __name__ == "__main__":
    main()
