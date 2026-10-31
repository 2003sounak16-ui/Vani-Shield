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
extra_df_unique = pd.DataFrame(extra_rows)

# Combine unique recordings first to prevent data leakage between train and val
combined_unique = pd.concat([df_kaggle_clean, df_multi_clean, extra_df_unique], ignore_index=True).drop_duplicates(subset=["path"])
print(f"Total unique master recordings: {len(combined_unique)}")

# Stratified split on unique recordings first
train_unique, val_unique = train_test_split(
    combined_unique,
    test_size=0.15,
    random_state=42,
    stratify=combined_unique["label"]
)
train_unique = train_unique.copy()
val_unique = val_unique.copy()
val_unique["split"] = "val"

# Oversample ONLY within the training set to prevent evaluation leakage
extra_sources = [
    "wa_ai_clone_rahul", "wa_ai_clone_sian", "wa_ai_clone_lottery",
    "user_real_voice", "real_friend_mobile"
]
train_extra = train_unique[train_unique["source"].isin(extra_sources)]
if not train_extra.empty:
    extra_boost = pd.concat([train_extra] * 24, ignore_index=True)
else:
    extra_boost = pd.DataFrame()

# Oversample Indic real data by 3x within training set only
train_indic_real = train_unique[(train_unique["source"] == "multilingual_cv") & (train_unique["label"] == 0)]
if not train_indic_real.empty:
    indic_real_boost = pd.concat([train_indic_real] * 2, ignore_index=True)
else:
    indic_real_boost = pd.DataFrame()

train_oversampled = pd.concat([train_unique, extra_boost, indic_real_boost], ignore_index=True)
train_oversampled["split"] = "train"

# Combine oversampled train set with clean unique validation set
combined = pd.concat([train_oversampled, val_unique], ignore_index=True)
print(f"Total master rows with training oversampling: {len(combined)}")
print("Class breakdown:\n", combined["label"].value_counts())
print("Source breakdown:\n", combined["source"].value_counts().head(10))

# Verify zero leakage between train and val
train_paths = set(train_oversampled["path"])
val_paths = set(val_unique["path"])
leakage = train_paths.intersection(val_paths)
assert len(leakage) == 0, f"Critical: Data leakage detected between train and val: {len(leakage)} overlapping paths!"
print(f"[Verification] Train/Val overlap: {len(leakage)} samples (Zero leakage guaranteed).")

output_path = "master_unified_manifest.csv"
combined.to_csv(output_path, index=False)
print(f"\n[OK] Wrote {len(combined)} rows to {output_path}")
print(combined.groupby(["split", "label"]).size())
