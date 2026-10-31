import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Kaggle 4447 Manifest (8 SOTA AI Clone Architectures + LibriSpeech Real)
df_kaggle = pd.read_csv("dataset_4447_manifest.csv")
print(f"Loaded Kaggle 4447 manifest: {len(df_kaggle)} rows")

# 2. Load Multilingual Manifest (Bengali, Hindi, Indian English Real + Fakes)
df_multi = pd.read_csv("multilingual_manifest.csv")
print(f"Loaded Multilingual manifest: {len(df_multi)} rows")

# Ensure matching schema
df_kaggle_clean = pd.DataFrame({
    "path": df_kaggle["path"],
    "label": df_kaggle["label"].astype(int),
    "language": df_kaggle["language"].fillna("en"),
    "source": df_kaggle["model_architecture"]
})

df_multi_clean = pd.DataFrame({
    "path": df_multi["path"],
    "label": df_multi["label"].astype(int),
    "language": df_multi["language"].fillna("indic"),
    "source": "multilingual_cv"
})

# 3. Add In-The-Wild WhatsApp AI Clones & Real User Voice
extra_rows = [
    # WhatsApp AI Clones
    {"path": r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.29.08 PM.ogg", "label": 1, "language": "en", "source": "wa_ai_clone_rahul"},
    {"path": r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.34.16 PM.ogg", "label": 1, "language": "en", "source": "wa_ai_clone_sian"},
    {"path": r"C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.05.18 PM.mpeg", "label": 1, "language": "en", "source": "wa_ai_clone_lottery"},
    # User Authentic Voice
    {"path": os.path.abspath("user_test_voice_16k.wav"), "label": 0, "language": "bn", "source": "user_real_voice"},
    {"path": os.path.abspath("data_demo/unseen_test/unseen_real_friend.wav"), "label": 0, "language": "en", "source": "real_friend_mobile"},
]

# Oversample the in-the-wild WhatsApp and user audio by 20x so they have sufficient gradient signal
extra_df = pd.DataFrame(extra_rows * 25)

# Also oversample Indic real data by 3x so Indian accents are strongly represented alongside LibriSpeech
indic_real = df_multi_clean[df_multi_clean["label"] == 0]
indic_real_boost = pd.concat([indic_real] * 3, ignore_index=True)

# Combine everything
combined = pd.concat([df_kaggle_clean, df_multi_clean, indic_real_boost, extra_df], ignore_index=True)
print(f"Total master samples: {len(combined)}")
print("Class breakdown:\n", combined["label"].value_counts())
print("Source breakdown:\n", combined["source"].value_counts().head(10))

# Stratified 80/20 train/val split
train_idx, val_idx = train_test_split(
    combined.index,
    test_size=0.15,
    random_state=42,
    stratify=combined["label"]
)

combined["split"] = "train"
combined.loc[val_idx, "split"] = "val"

output_path = "master_unified_manifest.csv"
combined.to_csv(output_path, index=False)
print(f"\n[OK] Wrote {len(combined)} rows to {output_path}")
print(combined.groupby(["split", "label"]).size())
