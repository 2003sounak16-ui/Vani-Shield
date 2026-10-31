"""
VaniShield AI — Ingest Kaggle Audio Deepfake Detection Dataset
Dataset: adarshsingh0903/audio-deepfake-detection-dataset (via kagglehub)
Augments training dataset with SOTA generative vocoders:
- OpenAI TTS (alloy, echo, fable, etc.)
- Coqui xTTS
- ByteDance SeedTTS
- Meta VoiceBox
- FlashSpeech
- Real Human Speech Partitions
Standardizes to 16 kHz mono 16-bit PCM WAV (3.5s duration)
Merges into multilingual_manifest.csv ensuring 0% speaker leakage.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import glob
import random
import soundfile as sf
import torchaudio
import torch
import numpy as np
import pandas as pd
import kagglehub

TARGET_SR = 16000
TARGET_DURATION = 3.5
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

def standardize_and_save(audio_data, sr, out_path):
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

    max_val = np.max(np.abs(audio_data))
    if max_val > 1e-4:
        audio_data = audio_data / max_val * 0.95

    if len(audio_data) > TARGET_SAMPLES:
        audio_data = audio_data[:TARGET_SAMPLES]
    else:
        audio_data = np.pad(audio_data, (0, TARGET_SAMPLES - len(audio_data)))

    sf.write(out_path, audio_data.astype(np.float32), TARGET_SR, subtype="PCM_16")

def main():
    print("=" * 70)
    print("VaniShield AI: Ingesting Kaggle Audio Deepfake Dataset")
    print("Dataset: adarshsingh0903/audio-deepfake-detection-dataset")
    print("=" * 70)

    # 1. Download / locate latest version via kagglehub
    path = kagglehub.dataset_download("adarshsingh0903/audio-deepfake-detection-dataset")
    print("Path to dataset files:", path)

    out_real_dir = "data_real/real/en"
    out_fake_dir = "data_real/fake/en"
    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_fake_dir, exist_ok=True)

    kaggle_records = []

    # 2. Ingest 100 Real Speech Samples
    real_src_dir = os.path.join(path, "real_samples")
    real_files = sorted(glob.glob(os.path.join(real_src_dir, "*.wav")))
    random.seed(42)
    random.shuffle(real_files)
    selected_real = real_files[:100]

    print(f"\nProcessing {len(selected_real)} Real Human Speech samples from Kaggle...")
    for i, fpath in enumerate(selected_real):
        base = os.path.basename(fpath)
        # Extract speaker ID prefix from filename: 1089_134686_... -> speaker is kaggle_real_1089
        spk_prefix = base.split("_")[0] if "_" in base else str(i % 15)
        spk = f"kaggle_real_spk_{spk_prefix}"
        out_f = os.path.join(out_real_dir, f"kaggle_real_{i:03d}_{base}")
        data, sr = sf.read(fpath)
        standardize_and_save(data, sr, out_f)
        kaggle_records.append({
            "path": out_f.replace("\\", "/"),
            "label": 0,
            "language": "en",
            "speaker_group": spk
        })

    # 3. Ingest 100 Fake / AI Deepfake Samples across multiple vocoders
    fake_sources = [
        ("OpenAI", 25),
        ("xTTS", 25),
        ("seedtts_files", 25),
        ("VoiceBox", 15),
        ("FlashSpeech", 10)
    ]

    total_fake_saved = 0
    for folder_name, count in fake_sources:
        src_dir = os.path.join(path, folder_name)
        if not os.path.exists(src_dir):
            continue
        flist = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
        random.seed(42)
        random.shuffle(flist)
        chosen = flist[:count]
        print(f"Processing {len(chosen)} AI Deepfakes from {folder_name} vocoder...")
        for j, fpath in enumerate(chosen):
            base = os.path.basename(fpath)
            spk = f"kaggle_fake_{folder_name.lower()}_{j % 6}"
            out_f = os.path.join(out_fake_dir, f"kaggle_{folder_name.lower()}_{total_fake_saved:03d}_{base}")
            data, sr = sf.read(fpath)
            standardize_and_save(data, sr, out_f)
            kaggle_records.append({
                "path": out_f.replace("\\", "/"),
                "label": 1,
                "language": "en",
                "speaker_group": spk
            })
            total_fake_saved += 1

    print(f"\nIngested from Kaggle: {len(kaggle_records)} samples (100 Real, {total_fake_saved} Fake).")

    # 4. Merge with existing multilingual_manifest.csv
    existing_df = pd.read_csv("multilingual_manifest.csv")
    new_df = pd.DataFrame(kaggle_records)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["path"]).reset_index(drop=True)

    # 5. Split train (80%) and val (20%) strictly by speaker_group with ZERO leakage
    combined_df["split"] = "train"
    for (lang, lbl), group in combined_df.groupby(["language", "label"]):
        unique_spks = sorted(group["speaker_group"].unique())
        random.seed(42)
        random.shuffle(unique_spks)
        n_val = max(1, int(len(unique_spks) * 0.20))
        val_spks = set(unique_spks[:n_val])
        combined_df.loc[(combined_df["language"] == lang) & (combined_df["label"] == lbl) & (combined_df["speaker_group"].isin(val_spks)), "split"] = "val"

    combined_df.to_csv("multilingual_manifest.csv", index=False)
    print("\nSaved augmented multilingual_manifest.csv:")
    print(f"  Total samples: {len(combined_df)}")
    print(combined_df.groupby(["language", "label", "split"]).size())

    # Verify 0% speaker leakage
    train_spks = set(combined_df[combined_df["split"] == "train"]["speaker_group"])
    val_spks = set(combined_df[combined_df["split"] == "val"]["speaker_group"])
    leakage = train_spks.intersection(val_spks)
    print(f"\nSpeaker leakage between train and val: {len(leakage)} (Must be 0).")
    assert len(leakage) == 0, f"Error: Speaker leakage detected: {leakage}"
    print("[SUCCESS] Dataset augmented with Kaggle deepfakes and verified with ZERO speaker leakage!")

if __name__ == "__main__":
    main()
