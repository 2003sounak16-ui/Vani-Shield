import os
import glob
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

print("Starting KaggleHub dataset download / verification...")
# 1. Download/locate dataset via kagglehub
dataset_dir = kagglehub.dataset_download("adarshsingh0903/audio-deepfake-detection-dataset")
print(f"[OK] Dataset cached at: {dataset_dir}")

flashspeech_dir = os.path.join(dataset_dir, "FlashSpeech")
real_dir = os.path.join(dataset_dir, "real_samples")

# Also check for case variations or subdirectories if not found directly
if not os.path.exists(flashspeech_dir):
    candidates = glob.glob(os.path.join(dataset_dir, "**", "FlashSpeech"), recursive=True)
    if candidates:
        flashspeech_dir = candidates[0]

if not os.path.exists(real_dir):
    candidates = glob.glob(os.path.join(dataset_dir, "**", "real_samples"), recursive=True)
    if candidates:
        real_dir = candidates[0]

print(f"FlashSpeech directory: {flashspeech_dir}")
print(f"Real samples directory: {real_dir}")

flashspeech_files = sorted(glob.glob(os.path.join(flashspeech_dir, "*.wav")))
real_files = sorted(glob.glob(os.path.join(real_dir, "*.wav")))[:len(flashspeech_files)]

print(f"Found {len(flashspeech_files)} FlashSpeech samples and {len(real_files)} paired Real samples.")

# 2. Build balanced dataframe
rows = []
for p in real_files:
    spk_id = os.path.basename(p).split("_")[0]
    rows.append({"path": p, "label": 0, "language": "en", "speaker_group": f"real_{spk_id}"})

for p in flashspeech_files:
    rows.append({"path": p, "label": 1, "language": "en", "speaker_group": "flashspeech"})

df = pd.DataFrame(rows)

# 3. 80/20 train/val split
train_df, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
train_df["split"] = "train"
val_df["split"] = "val"

manifest_df = pd.concat([train_df, val_df]).reset_index(drop=True)
output_csv = "flashspeech_manifest.csv"
manifest_df.to_csv(output_csv, index=False)
print(f"[OK] Saved {output_csv}: {len(train_df)} train, {len(val_df)} val (Total: {len(manifest_df)}).")
