import os, sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
out_manifest = os.environ.get("VANISHIELD_OUT_MANIFEST", os.path.join(PROJECT_ROOT, 'master_unified_dataset_manifest.csv'))
balanced_path = os.environ.get("VANISHIELD_BALANCED_MANIFEST", os.path.join(PROJECT_ROOT, 'master_balanced_manifest.csv'))
wild_path = os.environ.get("VANISHIELD_WILD_MANIFEST", os.path.join(PROJECT_ROOT, 'data_real', 'in_the_wild_manifest.csv'))

# 1. Existing balanced dataset
if os.path.exists(balanced_path):
    df_balanced = pd.read_csv(balanced_path)
else:
    df_balanced = pd.DataFrame(columns=['path', 'label', 'language', 'source'])
print(f"Loaded master_balanced_manifest: {len(df_balanced)} rows")

# 2. In-The-Wild curated modern celebrity deepfakes & bona-fide speeches
if os.path.exists(wild_path):
    df_wild = pd.read_csv(wild_path)
else:
    df_wild = pd.DataFrame(columns=['path', 'label', 'language', 'source'])
print(f"Loaded in_the_wild_manifest: {len(df_wild)} rows")

# 3. Additional multilingual samples from data_real
extra_records = []
for d, lbl in [('data_real/fake', 1), ('data_real/real', 0)]:
    full_d = os.path.join(PROJECT_ROOT, d)
    if not os.path.exists(full_d):
        continue
    for root, _, files in os.walk(full_d):
        if 'in_the_wild' in root:
            continue
        for f in files:
            if f.endswith('.wav'):
                p = os.path.abspath(os.path.join(root, f))
                lang = 'bn' if ('\\bn' in root or '/bn' in root) else ('hi' if ('\\hi' in root or '/hi' in root) else 'en')
                extra_records.append({
                    'path': p,
                    'label': lbl,
                    'language': lang,
                    'source': 'data_real_' + ('fake' if lbl == 1 else 'real')
                })

# Initialize DataFrame with explicit columns so KeyError is avoided when extra_records is empty
df_extra = pd.DataFrame(extra_records, columns=['path', 'label', 'language', 'source'])

# Normalize both sides to absolute canonical paths for deduplication
if not df_balanced.empty and 'path' in df_balanced.columns:
    existing_paths = set(os.path.normcase(os.path.abspath(p)) for p in df_balanced['path'].dropna())
else:
    existing_paths = set()

if not df_extra.empty and 'path' in df_extra.columns:
    df_extra_clean = df_extra[~df_extra['path'].apply(lambda p: os.path.normcase(os.path.abspath(p)) in existing_paths)].copy()
else:
    df_extra_clean = pd.DataFrame(columns=['path', 'label', 'language', 'source'])
print(f"Loaded extra data_real samples (non-duplicate): {len(df_extra_clean)} rows")

# 4. Recent test audio files
recent_samples = [
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 9.25.18 PM.mpeg', 1, 'en', 'recent_ai_qa_clone'),
    (r'C:\Users\souna\Downloads\WhatsApp Ptt 2026-10-30 at 9.28.19 PM.ogg', 0, 'bn', 'recent_wa_ptt_real'),
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 9.32.38 PM.ogg', 0, 'bn', 'recent_wa_makaut_real'),
    (r'C:\Users\souna\Downloads\WhatsApp Audio 2026-10-30 at 7.05.18 PM.mpeg', 1, 'bn', 'recent_lottery_scam'),
]
recent_records = []
for p, lbl, lang, src in recent_samples:
    if os.path.exists(p):
        recent_records.append({
            'path': os.path.abspath(p),
            'label': lbl,
            'language': lang,
            'source': src
        })
df_recent = pd.DataFrame(recent_records, columns=['path', 'label', 'language', 'source'])
print(f"Loaded recent WhatsApp samples: {len(df_recent)} rows")

# 5. Combine and shuffle
frames_to_concat = [f for f in [df_balanced, df_wild, df_extra_clean, df_recent] if not f.empty]
if frames_to_concat:
    combined_df = pd.concat(frames_to_concat, ignore_index=True)
else:
    combined_df = pd.DataFrame(columns=['path', 'label', 'language', 'source'])

combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 6. Verify audio file paths and report missing files
if not combined_df.empty and 'path' in combined_df.columns:
    print("Verifying audio file paths...")
    valid_mask = combined_df['path'].apply(os.path.exists)
    missing_count = int((~valid_mask).sum())
    if missing_count > 0:
        print(f"[WARN] {missing_count} rows have missing audio paths on disk.")
    combined_df = combined_df[valid_mask].reset_index(drop=True)

os.makedirs(os.path.dirname(os.path.abspath(out_manifest)), exist_ok=True)
combined_df.to_csv(out_manifest, index=False)
print("\n" + "=" * 60)
print(f"MASTER UNIFIED DATASET MANIFEST CREATED: {out_manifest}")
print(f"Total valid samples: {len(combined_df)}")
if not combined_df.empty and 'label' in combined_df.columns:
    print("\nLabel distribution:")
    print(combined_df['label'].value_counts())
if not combined_df.empty and 'source' in combined_df.columns:
    print("\nSource breakdown:")
    print(combined_df['source'].value_counts())
print("=" * 60)
