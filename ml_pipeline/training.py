import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for Voice Cloning Detection on RTX 3050")
    parser.add_argument("--manifest", type=str, default="pilot_manifest.csv")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", type=str, default="./results_pilot")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_duration", type=float, default=3.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--augment", action="store_true", help="Apply telephone/codec-like audio augmentation to training clips")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume, including optimizer and scheduler state.",
    )
    return parser.parse_args()

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor, max_duration=3.0, target_sr=16000, augment=False):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.max_samples = int(max_duration * target_sr)
        self.resamplers = {}
        self.augment = augment

    def _augment(self, waveform):
        """Approximate phone/codecs without changing the class label."""
        if torch.rand(1).item() < 0.6:
            # Narrowband telephone speech (8 kHz) returned to model rate.
            waveform = torchaudio.functional.resample(waveform, self.target_sr, 8000)
            waveform = torchaudio.functional.resample(waveform, 8000, self.target_sr)
        if torch.rand(1).item() < 0.7:
            rms = waveform.pow(2).mean().sqrt().clamp_min(1e-5)
            waveform = waveform + torch.randn_like(waveform) * rms * float(torch.empty(1).uniform_(0.002, 0.02))
        if torch.rand(1).item() < 0.7:
            waveform = waveform * float(torch.empty(1).uniform_(0.6, 1.4))
        return waveform.clamp(-1.0, 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["path"])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self.resamplers[sr](waveform)
        if self.augment:
            waveform = self._augment(waveform)

        waveform = waveform.squeeze(0).numpy()
        if len(waveform) > self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        inputs = self.feature_extractor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["label"] = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def main():
    args = parse_args()
    print("--- VaniShield AI: GPU Pilot Training Smoke Test ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(args.manifest)
    label_map = {"real": 0, "human": 0, "genuine": 0, "fake": 1, "synthetic": 1, "spoof": 1}
    if df["label"].dtype == object:
        df["label_idx"] = df["label"].astype(str).str.strip().str.lower().map(label_map)
    else:
        df["label_idx"] = df["label"].astype(int)

    if "split" in df.columns and "train" in df["split"].values and "val" in df["split"].values:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label_idx"])

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        label2id={"real": 0, "fake": 1},
        id2label={0: "real", 1: "fake"}
    )
    # Freeze convolutional feature encoder to stay strictly within 6GB VRAM
    model.freeze_feature_encoder()

    train_ds = AudioDataset(train_df, feature_extractor, max_duration=args.max_duration, augment=args.augment)
    val_ds = AudioDataset(val_df, feature_extractor, max_duration=args.max_duration)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=args.logging_steps,
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

    if args.resume_from_checkpoint:
        print(f"\nResuming training from: {args.resume_from_checkpoint}")
    else:
        print("\nStarting training on RTX 3050...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Pilot Training Pass Successful!")

    eval_results = trainer.evaluate()
    print("\nPilot Evaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    # Save initial pilot weights
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
