import zipfile, io, os, sys, soundfile as sf, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

zip_path = r'data_external\in_the_wild\release_in_the_wild.zip'
out_dir = r'data_real\in_the_wild'
os.makedirs(out_dir, exist_ok=True)

zf = zipfile.ZipFile(zip_path)
with zf.open('release_in_the_wild/meta.csv') as f:
    df = pd.read_csv(f)

modern_speakers = [
    'Barack Obama', 'Donald Trump', 'Bernie Sanders', 'Mark Zuckerberg',
    'Adam Driver', 'Arnold Schwarzenegger', 'Bill Gates', 'Elon Musk',
    'Morgan Freeman', 'Joe Biden'
]

modern_df = df[df['speaker'].isin(modern_speakers)].copy()

# Sample 500 spoofs and 500 bona-fide
spoof_sample = modern_df[modern_df['label'] == 'spoof'].sample(500, random_state=42)
real_sample = modern_df[modern_df['label'] == 'bona-fide'].sample(500, random_state=42)

selected = pd.concat([spoof_sample, real_sample]).sample(frac=1.0, random_state=42)
print(f"Extracting {len(selected)} curated samples (500 spoof, 500 real) to {out_dir}...")

records = []
for idx, row in selected.iterrows():
    in_zip_name = 'release_in_the_wild/' + row['file']
    out_file = os.path.join(out_dir, row['file'])
    
    # Extract
    with zf.open(in_zip_name) as f_in, open(out_file, 'wb') as f_out:
        f_out.write(f_in.read())
        
    records.append({
        'path': os.path.abspath(out_file),
        'label': 1 if row['label'] == 'spoof' else 0,
        'language': 'en',
        'source': 'InTheWild_' + ('spoof' if row['label'] == 'spoof' else 'real')
    })

manifest_df = pd.DataFrame(records)
manifest_df.to_csv(r'data_real\in_the_wild_manifest.csv', index=False)
print(f"Extraction complete! Saved manifest to data_real\\in_the_wild_manifest.csv ({len(manifest_df)} rows).")
total_size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / (1024*1024)
print(f"Total extracted folder size: {total_size:.2f} MB")
