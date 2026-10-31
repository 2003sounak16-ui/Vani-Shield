import argparse
import time
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import numpy as np

def load_audio(audio_path):
    """Load common audio formats, including M4A/AAC when libsndfile lacks support."""
    try:
        return torchaudio.load(audio_path)
    except Exception as torchaudio_error:
        try:
            import av

            container = av.open(audio_path)
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="mono", rate=16000
            )
            chunks = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None) or []:
                chunks.append(resampled.to_ndarray().reshape(-1))
            container.close()
            if not chunks:
                raise ValueError("No decodable audio frames found")
            return torch.from_numpy(np.concatenate(chunks)).unsqueeze(0), 16000
        except Exception as av_error:
            raise RuntimeError(
                f"Unable to decode audio with torchaudio or PyAV: {audio_path}"
            ) from av_error

def predict(audio_path, model_dir="results_pilot", max_duration=3.0, num_windows=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Load and resample audio to 16kHz mono
    waveform, sr = load_audio(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    # Score evenly distributed windows rather than trusting only the first
    # three seconds. This reduces errors caused by silence, introductions, or
    # short-lived codec artifacts at the start of a call recording.
    max_samples = int(max_duration * 16000)
    waveform = waveform.squeeze(0).numpy()
    if len(waveform) <= max_samples:
        starts = [0]
    else:
        starts = np.linspace(0, len(waveform) - max_samples, num=max(1, num_windows), dtype=int).tolist()

    window_probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in starts:
            clip = waveform[start:start + max_samples]
            if len(clip) < max_samples:
                clip = np.pad(clip, (0, max_samples - len(clip)))
            inputs = feature_extractor(clip, sampling_rate=16000, return_tensors="pt").to(device)
            outputs = model(**inputs)
            window_probs.append(torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy())
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = np.mean(window_probs, axis=0)

    labels = {0: "REAL / GENUINE", 1: "FAKE / AI-CLONED"}
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    print(f"\n--- Prediction for: {audio_path} ---")
    print(f"Result     : {labels[pred_idx]}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Real Prob  : {probs[0] * 100:.2f}%")
    print(f"Fake Prob  : {probs[1] * 100:.2f}%")
    print(f"Windows    : {len(window_probs)} x {max_duration:.1f}s")
    print(f"Fake range : {min(p[1] for p in window_probs) * 100:.2f}% - {max(p[1] for p in window_probs) * 100:.2f}%")
    print(f"Latency    : {latency_ms:.1f} ms on {device.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to .wav file")
    parser.add_argument("--model_dir", type=str, default="results_pilot")
    parser.add_argument("--num_windows", type=int, default=5, help="Number of evenly spaced audio windows to average")
    args = parser.parse_args()
    predict(args.audio_path, args.model_dir, num_windows=args.num_windows)
