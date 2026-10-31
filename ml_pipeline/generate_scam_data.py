"""
VaniShield AI — Neural AI Scam Audio Generator
Team CypherEye | Project VaniShield AI
Synthesizes neural AI voice cloning vishing attacks in English, Hindi, and Bengali using edge-tts.
"""

import os
import sys
import asyncio
import soundfile as sf
import librosa

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_samples", "fake")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = [
    {
        "filename": "scam_sbi_otp.wav",
        "lang": "en",
        "voice": "en-IN-PrabhatNeural",
        "text": "Urgent security notice from State Bank of India fraud prevention center. Your account has an unauthorized debit of sixty-five thousand rupees. Share your six-digit OTP immediately to block the transaction."
    },
    {
        "filename": "scam_card_cvv.wav",
        "lang": "en",
        "voice": "en-IN-NeerjaNeural",
        "text": "Dear customer, your credit card rewards of fifteen thousand rupees have expired. Please verify your sixteen-digit card number and three-digit CVV on the back to claim your balance directly."
    },
    {
        "filename": "scam_kyc_hindi.wav",
        "lang": "hi",
        "voice": "hi-IN-MadhurNeural",
        "text": "नमस्कार, बैंक हेड ऑफिस से बोल रहे हैं। आपके खाते का ईकेवाईसी निलंबित हो चुका है। तुरंत अपना आधार नंबर और पंजीकृत मोबाइल पर आया हुआ छह अंकों का ओटीपी बताएं अन्यथा खाता हमेशा के लिए सीज कर दिया जाएगा।"
    },
    {
        "filename": "scam_upi_pin.wav",
        "lang": "en",
        "voice": "en-IN-PrabhatNeural",
        "text": "Hello, I am calling from payment gateway technical support. A pending refund of four thousand five hundred rupees is waiting in your account. Open Google Pay right now and enter your UPI PIN to receive the funds."
    },
    {
        "filename": "scam_police_bn.wav",
        "lang": "bn",
        "voice": "bn-IN-BashkarNeural",
        "text": "জরুরি সতর্কতা: আমি কলকাতা পুলিশ সাইবার ক্রাইম ব্রাঞ্চ থেকে অফিসার বলছি। আপনার আধার কার্ডের মাধ্যমে আন্তর্জাতিক অর্থ পাচার হয়েছে। গ্রেফতারি এড়াতে এখনই আপনার ব্যাংক বিবরণ ও ওটিপি যাচাই করুন।"
    }
]

async def synthesize_scenario(item):
    if not EDGE_TTS_AVAILABLE:
        print("[ERROR] edge-tts package is required. Install via: pip install edge-tts")
        return

    mp3_tmp = os.path.join(OUTPUT_DIR, f"temp_{item['filename']}.mp3")
    wav_target = os.path.join(OUTPUT_DIR, item["filename"])

    print(f"[*] Generating: {item['filename']} [{item['voice']}]...")
    comm = edge_tts.Communicate(item["text"], item["voice"])
    await comm.save(mp3_tmp)

    # Convert to 16kHz mono WAV for ML pipelines
    y, sr = librosa.load(mp3_tmp, sr=16000, mono=True)
    sf.write(wav_target, y, 16000)

    if os.path.exists(mp3_tmp):
        os.remove(mp3_tmp)

    print(f"  [OK] Saved 16kHz WAV: {wav_target} ({len(y)/16000:.1f}s)")

async def main():
    print("\n" + "=" * 70)
    print(" VANISHIELD AI -- NEURAL SCAM AUDIO GENERATOR")
    print(f" Output Target: {OUTPUT_DIR}")
    print("=" * 70)

    for scenario in SCENARIOS:
        try:
            await synthesize_scenario(scenario)
        except Exception as e:
            print(f"  [!] Failed to generate {scenario['filename']}: {e}")

    print("\n[SUCCESS] Scam audio dataset generation completed.\n")

if __name__ == "__main__":
    asyncio.run(main())
