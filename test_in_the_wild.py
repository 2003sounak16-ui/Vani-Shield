import zipfile, io, soundfile as sf, pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests

url = 'http://127.0.0.1:8000/api/evaluate/upload'

zf = zipfile.ZipFile(r'data_external\in_the_wild\release_in_the_wild.zip')
with zf.open('release_in_the_wild/meta.csv') as f:
    df = pd.read_csv(f)

spoofs = df[df['label'] == 'spoof'].sample(10, random_state=42)
reals = df[df['label'] == 'bona-fide'].sample(10, random_state=42)

print('=== 10 SPOOFS FROM IN-THE-WILD ===')
spoof_passed = 0
for _, row in spoofs.iterrows():
    fname = 'release_in_the_wild/' + row['file']
    audio_bytes = zf.read(fname)
    resp = requests.post(url, files={'file': (row['file'], audio_bytes, 'audio/wav')})
    res = resp.json()
    w2v = res.get('w2v_fake_prob', 0)
    risk = res.get('unified_risk_score', 0)
    threat = res.get('threat_type')
    correct = (threat != 'AUTHENTIC_SAFE_CALL')
    if correct:
        spoof_passed += 1
    print(f"Spoof {row['file']} ({row['speaker']}): Risk={risk}%, W2V={w2v}%, Threat={threat} -> {'PASS' if correct else 'FAIL'}")

print('\n=== 10 BONA-FIDE (REAL) FROM IN-THE-WILD ===')
real_passed = 0
for _, row in reals.iterrows():
    fname = 'release_in_the_wild/' + row['file']
    audio_bytes = zf.read(fname)
    resp = requests.post(url, files={'file': (row['file'], audio_bytes, 'audio/wav')})
    res = resp.json()
    w2v = res.get('w2v_fake_prob', 0)
    risk = res.get('unified_risk_score', 0)
    threat = res.get('threat_type')
    correct = (threat == 'AUTHENTIC_SAFE_CALL')
    if correct:
        real_passed += 1
    print(f"Real {row['file']} ({row['speaker']}): Risk={risk}%, W2V={w2v}%, Threat={threat} -> {'PASS' if correct else 'FAIL'}")

print(f"\nSpoof accuracy: {spoof_passed}/10 ({spoof_passed*10}%)")
print(f"Real accuracy: {real_passed}/10 ({real_passed*10}%)")
print(f"Total zero-shot accuracy: {(spoof_passed+real_passed)}/20 ({(spoof_passed+real_passed)*5}%)")
