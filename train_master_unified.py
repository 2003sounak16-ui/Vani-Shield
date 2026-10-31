import os, sys, argparse, pandas as pd, numpy as np, torch, torchaudio, av, soundfile as sf, librosa
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)

def load_audio_robust(path, target_sr=16000):
    # 1. Fast path: torchaudio
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        return waveform
    except Exception:
        pass

    # 2. Universal media path: PyAV (Opus, Ogg, Mpeg, M4a, AAC, Flac)
    try:
        container = av.open(path)
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
        frames = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray())
        if frames:
            arr = np.concatenate(frames, axis=1).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    except Exception:
        pass

    # 3. Fallback: soundfile / librosa
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    except Exception as e:
        # Return 1 second of silence as last resort to avoid crash
        return torch.zeros((1, target_sr), dtype=torch.float32)

class UnifiedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor, max_duration=3.0, target_sr=16000, augment=False):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.max_samples = int(max_duration * target_sr)
        self.augment = augment

    def _augment(self, waveform):
        if torch.rand(1).item() < 0.5:
            waveform = torchaudio.functional.resample(waveform, self.target_sr, 8000)
            waveform = torchaudio.functional.resample(waveform, 8000, self.target_sr)
        if torch.rand(1).item() < 0.6:
            rms = waveform.pow(2).mean().sqrt().clamp_min(1e-5)
            waveform = waveform + torch.randn_like(waveform) * rms * float(torch.empty(1).uniform_(0.002, 0.02))
        if torch.rand(1).item() < 0.6:
            waveform = waveform * float(torch.empty(1).uniform_(0.7, 1.3))
        return waveform.clamp(-1.0, 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = load_audio_robust(row["path"], target_sr=self.target_sr)
        if self.augment:
            waveform = self._augment(waveform)

        waveform = waveform.squeeze(0).numpy()
        if len(waveform) > self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        inputs = self.feature_extractor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def main():
    print("=" * 70)
    print("VaniShield AI: Unified Master Dataset Fine-Tuning")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    manifest_path = r"C:\AI\VoiceCloning\master_unified_dataset_manifest.csv"
    output_dir = r"C:\AI\VoiceCloning\results_master_unified"
    base_model = r"C:\AI\VoiceCloning\results_master_balanced"

    df = pd.read_csv(manifest_path)
    print(f"Dataset total samples: {len(df)}")
    print(f"Base model checkpoint: {base_model}")
    print(f"Output directory: {output_dir}")

    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    print(f"Train split: {len(train_df)}, Validation split: {len(val_df)}")

    fe = AutoFeatureExtractor.from_pretrained(base_model)
    model = AutoModelForAudioClassification.from_pretrained(
        base_model,
        num_labels=2,
        label2id={"real": 0, "fake": 1},
        id2label={0: "real", 1: "fake"}
    )
    model.freeze_feature_encoder()

    train_ds = UnifiedAudioDataset(train_df, fe, max_duration=3.0, augment=True)
    val_ds = UnifiedAudioDataset(val_df, fe, max_duration=3.0, augment=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("\nStarting Training on RTX 3050...")
    trainer.train()

    print("\nEvaluating Model Performance...")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    trainer.save_model(output_dir)
    fe.save_pretrained(output_dir)
    print(f"\nModel successfully saved to: {output_dir}")

if __name__ == "__main__":
    main()
