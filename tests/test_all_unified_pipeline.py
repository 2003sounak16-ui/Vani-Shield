import sys, os, glob
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from faster_whisper import WhisperModel
import numpy as np
import re

cp = r'C:\AI\VoiceCloning\results_master_unified'
model = AutoModelForAudioClassification.from_pretrained(cp).to('cuda')
fe = AutoFeatureExtractor.from_pretrained(cp)
model.eval()

asr = WhisperModel('base', device='cpu', compute_type='int8')

SCAM_PATTERNS = [
    r'\b(?:otp|one\s*time\s*password|pin|cvv|password)\b',
    r'\b(?:bank|rbi|sbi|hdfc|icici|axis|punjab\s*national|paytm|phonepe|gpay)\b',
    r'\b(?:kyc|pan\s*card|aadhaar|update|verify|verification|expire|block)\b',
    r'\b(?:transfer|send|urgent|immediate|police|cbi|arrest|court|customs)\b',
    r'\b(?:winner|lottery|prize|reward|congratulations|crore|lakh)\b',
    r'\b(?:boss|manager|ceo|send\s*(?:it\s*)?to\s*me|give\s*me\s*rs|account\s*number)\b',
    r'\b(?:बरानज|बरान्च|शाखा|पा़ेसे|पैसे|खाता|रुपये|ले\s*लो|लेरो|بینک|اکاؤنٹ|پیسہ)\b'
]

def evaluate(path):
    y, sr = librosa.load(path, sr=16000)
    dur = len(y) / sr
    
    # 1. Acoustic with 1.5s windows
    win_size = int(1.5 * sr)
    hop_size = int(0.75 * sr)
    probs = []
    for s in range(0, max(1, len(y) - win_size + 1), hop_size):
        c = y[s : s + win_size]
        if np.sqrt(np.mean(c**2)) < 0.012: continue
        inputs = fe(c, sampling_rate=16000, return_tensors='pt')['input_values'].to('cuda')
        with torch.no_grad():
            l = model(inputs).logits
            p = torch.softmax(l, dim=-1)[0, 1].item() * 100.0
            probs.append(p)
            
    if not probs:
        p_max, p_med, p_75, high_ratio = 0.0, 0.0, 0.0, 0.0
    else:
        p_max = float(np.max(probs))
        p_med = float(np.median(probs))
        p_75 = float(np.percentile(probs, 75))
        high_cnt = sum(1 for p in probs if p >= 75.0)
        high_ratio = high_cnt / len(probs)
        
    # 2. ASR across full audio (up to 90s)
    segs, info = asr.transcribe(
        y[:16000*90],
        beam_size=3,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        condition_on_previous_text=False
    )
    transcript = ' '.join([s.text for s in segs]).strip()
    
    flagged = []
    for pat in SCAM_PATTERNS:
        matches = re.findall(pat, transcript, flags=re.IGNORECASE)
        if matches: flagged.extend(matches)
    flagged = list(set(flagged))
    
    # 3. Intent scoring
    if len(flagged) >= 2:
        jev_p = min(99.0, 85.0 + len(flagged) * 5.0)
    elif len(flagged) == 1:
        jev_p = 80.0
    else:
        jev_p = 0.0
        
    # 4. Forensic Acoustic Arbitration
    # If Jev flags scam AND baseline is mostly human (p_med <= 2.0% and high_ratio < 0.35)
    if jev_p >= 75.0 and p_med <= 2.0 and high_ratio < 0.35:
        w2v_p = 0.1
        threat = 'FINANCIAL_SCAM (HUMAN FRAUDSTER)'
    elif p_max >= 95.0 or (high_ratio >= 0.25 and p_75 >= 50.0) or (p_med >= 30.0):
        w2v_p = min(99.9, max(p_75, p_max))
        threat = 'HYBRID_AI_VOICE_SCAM' if jev_p >= 70.0 else 'AI_VOICE_CLONE'
    elif p_max > 25.0 and high_ratio >= 0.10:
        w2v_p = p_75
        threat = 'HYBRID_AI_VOICE_SCAM' if jev_p >= 70.0 else 'AI_VOICE_CLONE'
    else:
        w2v_p = round(p_med, 1)
        threat = 'FINANCIAL_SCAM (HUMAN FRAUDSTER)' if jev_p >= 70.0 else 'AUTHENTIC_SAFE_CALL'
        
    return {
        'dur': round(dur, 1),
        'w2v': round(w2v_p, 1),
        'jev': round(jev_p, 1),
        'threat': threat,
        'flagged': flagged,
        'transcript': transcript[:60]
    }

for f in sorted(glob.glob(r'C:\AI\VoiceCloning\audio\*.*')):
    fn = os.path.basename(f)
    try:
        res = evaluate(f)
        print(f"{fn:45s} -> W2V: {res['w2v']:5.1f}%, Jev: {res['jev']:5.1f}%, Threat: {res['threat']}")
        if res['flagged']:
            print(f"   [!] Flagged: {res['flagged']}")
    except Exception as e:
        print(f"{fn:45s} -> ERROR: {e}")
