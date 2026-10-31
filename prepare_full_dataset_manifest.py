import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_dir = r"C:\Users\souna\.cache\kagglehub\datasets\adarshsingh0903\audio-deepfake-detection-dataset\versions\1"

model_dirs = {
    "real_samples": {"label": 0, "name": "real"},
    "FlashSpeech": {"label": 1, "name": "FlashSpeech"},
    "NaturalSpeech3": {"label": 1, "name": "NaturalSpeech3"},
    "OpenAI": {"label": 1, "name": "OpenAI"},
    "PromptTTS2": {"label": 1, "name": "PromptTTS2"},
    "seedtts_files": {"label": 1, "name": "SeedTTS"},
    "VALLE": {"label": 1, "name": "VALL-E"},
    "VoiceBox": {"label": 1, "name": "VoiceBox"},
    "xTTS": {"label": 1, "name": "XTTS"}
}

rows = []
for folder, meta in model_dirs.items():
    folder_path = os.path.join(dataset_dir, folder)
    wav_files = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
    for p in wav_files:
        rows.append({
            "path": p,
            "label": meta["label"],
            "model_architecture": meta["name"],
            "language": "en"
        })

df = pd.DataFrame(rows)
print(f"Total dataset samples collected: {len(df)}")
print("\nClass breakdown:")
print(df.groupby(["label", "model_architecture"]).size())

# Stratified 80/20 train/val split by model_architecture to ensure balanced representation across all 8 AI models + real
train_dfs = []
val_dfs = []

for model_name, group in df.groupby("model_architecture"):
    if len(group) >= 2:
        tr, va = train_test_split(group, test_size=0.20, random_state=42)
    else:
        tr = group
        va = group
    tr = tr.copy()
    va = va.copy()
    tr["split"] = "train"
    va["split"] = "val"
    train_dfs.append(tr)
    val_dfs.append(va)

manifest_df = pd.concat([pd.concat(train_dfs), pd.concat(val_dfs)]).reset_index(drop=True)
output_file = "dataset_4447_manifest.csv"
manifest_df.to_csv(output_file, index=False)

train_count = len(manifest_df[manifest_df['split'] == 'train'])
val_count = len(manifest_df[manifest_df['split'] == 'val'])
print(f"\n[OK] Successfully generated {output_file}:")
print(f"  - Total samples : {len(manifest_df)}")
print(f"  - Train split   : {train_count}")
print(f"  - Val split     : {val_count}")
