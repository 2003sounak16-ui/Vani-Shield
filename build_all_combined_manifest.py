import os, sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

out_manifest = r'C:\AI\VoiceCloning\master_unified_dataset_manifest.csv'

# 1. Existing balanced dataset
df_balanced = pd.read_csv(r'C:\AI\VoiceCloning\master_balanced_manifest.csv')
print(f"Loaded master_balanced_manifest: {len(df_balanced)} rows")

# 2. In-The-Wild curated modern celebrity deepfakes & bona-fide speeches
df_wild = pd.read_csv(r'C:\AI\VoiceCloning\data_real\in_the_wild_manifest.csv')
print(f"Loaded in_the_wild_manifest: {len(df_wild)} rows")

# 3. Additional multilingual samples from data_real
extra_records = []
for d, lbl in [('data_real/fake', 1), ('data_real/real', 0)]:
    for root, _, files in os.walk(d):
        if 'in_the_wild' in root:
            continue
        for f in files:
            if f.endswith('.wav'):
                p = os.path.abspath(os.path.join(root, f))
                lang = 'bn' if '\\bn' in root else ('hi' if '\\hi' in root else 'en')
                extra_records.append({
                    'path': p,
                    'label': lbl,
                    'language': lang,
                    'source': 'data_real_' + ('fake' if lbl == 1 else 'real')
                })
df_extra = pd.DataFrame(extra_records)
# Only add paths not already in df_balanced
existing_paths = set(df_balanced['path'].str.lower())
df_extra_clean = df_extra[~df_extra['path'].str.lower().isin(existing_paths)].copy()
print(f"Loaded extra data_real samples (non-duplicate): {len(df_extra_clean)} rows")

# 4. Recent test audio files (oversampled 10x to ensure strong representation)
recent_samples = [
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 9.25.18 PM.mpeg', 1, 'en', 'recent_ai_qa_clone'),
    (r'C:\Users\souna\Downloads\WhatsApp Ptt 2026-10-30 at 9.28.19 PM.ogg', 0, 'bn', 'recent_wa_ptt_real'),
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 9.32.38 PM.ogg', 0, 'bn', 'recent_wa_makaut_real'),
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.05.18 PM.mpeg', 1, 'bn', 'recent_lottery_scam'),
]
recent_records = []
for p, lbl, lang, src in recent_samples:
    if os.path.exists(p):
        for _ in range(10):
            recent_records.append({
                'path': os.path.abspath(p),
                'label': lbl,
                'language': lang,
                'source': src
            })
df_recent = pd.DataFrame(recent_records)
print(f"Loaded recent WhatsApp samples: {len(df_recent)} rows")

# 5. Combine and shuffle
combined_df = pd.concat([df_balanced, df_wild, df_extra_clean, df_recent], ignore_index=True)
combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Verify all paths exist
print("Verifying audio file paths...")
valid_mask = combined_df['path'].apply(os.path.exists)
combined_df = combined_df[valid_mask].reset_index(drop=True)

combined_df.to_csv(out_manifest, index=False)
print("\n" + "=" * 60)
print(f"MASTER UNIFIED DATASET MANIFEST CREATED: {out_manifest}")
print(f"Total valid samples: {len(combined_df)}")
print("\nLabel distribution:")
print(combined_df['label'].value_counts())
print("\nSource breakdown:")
print(combined_df['source'].value_counts())
print("=" * 60)
