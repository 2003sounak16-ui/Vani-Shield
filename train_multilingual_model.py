"""
VaniShield AI — Multilingual Wav2Vec 2.0 Voice Cloning Detection Model
Targets: English (en), Bengali (bn), Hindi (hi)
Trained on RTX 3050 6GB GPU with FP16, frozen feature encoder, and strict speaker-disjoint validation.
Outputs: Accuracy, Precision, Recall, F1, and Equal Error Rate (EER) per language and overall.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate (EER) where FPR == FNR."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find point where FPR == FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor, max_duration=3.5, target_sr=16000):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.max_samples = int(max_duration * target_sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["path"]
        try:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
            waveform = waveform.squeeze(0).numpy()
        except Exception:
            waveform = np.zeros(self.max_samples, dtype=np.float32)

        if len(waveform) > self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        inputs = self.feature_extractor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item

def evaluate_multilingual(model, feature_extractor, val_df, device, max_duration=3.5):
    """Detailed evaluation broken down by language: en, bn, hi."""
    model.eval()
    val_ds = AudioDataset(val_df, feature_extractor, max_duration=max_duration)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_values=input_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_df_eval = val_df.copy().reset_index(drop=True)
    val_df_eval["pred"] = all_preds
    val_df_eval["prob_fake"] = all_probs
    val_df_eval["true_label"] = all_labels

    print("\n" + "=" * 70)
    print("VaniShield AI: Multilingual Evaluation Metrics (Speaker-Disjoint)")
    print("=" * 70)

    # 1. Overall Metrics
    overall_acc = accuracy_score(val_df_eval["true_label"], val_df_eval["pred"])
    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        val_df_eval["true_label"], val_df_eval["pred"], average="binary", zero_division=0
    )
    overall_eer = compute_eer(val_df_eval["true_label"], val_df_eval["prob_fake"])

    print(f"OVERALL RESULTS:")
    print(f"  Accuracy:  {overall_acc * 100:.2f}%")
    print(f"  Precision: {overall_p * 100:.2f}%")
    print(f"  Recall:    {overall_r * 100:.2f}%")
    print(f"  F1-Score:  {overall_f1 * 100:.2f}%")
    print(f"  EER:       {overall_eer * 100:.2f}%")
    print(f"  Confusion Matrix:\n{confusion_matrix(val_df_eval['true_label'], val_df_eval['pred'])}")

    # 2. Per-Language Breakdown
    print("\nPER-LANGUAGE BREAKDOWN:")
    lang_metrics = {}
    for lang in ["en", "bn", "hi"]:
        sub = val_df_eval[val_df_eval["language"] == lang]
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub["true_label"], sub["pred"])
        p, r, f1, _ = precision_recall_fscore_support(
            sub["true_label"], sub["pred"], average="binary", zero_division=0
        )
        eer = compute_eer(sub["true_label"], sub["prob_fake"])
        lang_names = {"en": "English", "bn": "Bengali (বাংলা)", "hi": "Hindi (हिन्दी)"}
        print(f"\n  --- Language: {lang_names.get(lang, lang)} ({len(sub)} validation samples) ---")
        print(f"    Accuracy:  {acc * 100:.2f}%")
        print(f"    Precision: {p * 100:.2f}%")
        print(f"    Recall:    {r * 100:.2f}%")
        print(f"    F1-Score:  {f1 * 100:.2f}%")
        print(f"    EER:       {eer * 100:.2f}%")
        cm = confusion_matrix(sub["true_label"], sub["pred"])
        print(f"    Confusion Matrix:\n{cm}")
        lang_metrics[lang] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "eer": eer}

    print("=" * 70)

    # 3. Per-Generator Breakdown (if available)
    if "generator" in val_df_eval.columns:
        print("\nPER-GENERATOR ACCURACY BREAKDOWN (FAKE SAMPLES):")
        fake_sub = val_df_eval[val_df_eval["true_label"] == 1]
        for gen, group in fake_sub.groupby("generator"):
            gen_acc = (group["pred"] == 1).mean() * 100
            print(f"  - {gen:16s} ({len(group):3d} samples): {gen_acc:6.2f}% Detected")

        real_sub = val_df_eval[val_df_eval["true_label"] == 0]
        real_acc = (real_sub["pred"] == 0).mean() * 100
        print(f"  - Genuine Human Speech ({len(real_sub):3d} samples): {real_acc:6.2f}% Correctly Identified")
        print("=" * 70)

    return {
        "overall": {"accuracy": overall_acc, "precision": overall_p, "recall": overall_r, "f1": overall_f1, "eer": overall_eer},
        "per_language": lang_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="multilingual_manifest.csv")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", type=str, default="./results_real_multilingual")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_duration", type=float, default=3.5)
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only using saved checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.eval_only:
        print("--- Running Multilingual Evaluation on Checkpoint ---")
        fe_path = args.output_dir if os.path.exists(os.path.join(args.output_dir, "preprocessor_config.json")) else args.model_name
        fe = AutoFeatureExtractor.from_pretrained(fe_path)
        eval_model = AutoModelForAudioClassification.from_pretrained(args.output_dir).to(device)
        val_df = pd.read_csv(args.manifest)
        val_df = val_df[val_df["split"] == "val"].reset_index(drop=True)
        metrics = evaluate_multilingual(eval_model, fe, val_df, device, max_duration=args.max_duration)
        return
    print("=" * 70)
    print("VaniShield AI: Multilingual Wav2Vec 2.0 Fine-Tuning")
    print(f"Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM Optimization: FP16=True | Batch={args.batch_size} | GradAccum={args.grad_accum} (Effective={args.batch_size*args.grad_accum})")
    print("=" * 70)

    # 1. Load Manifest
    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"\nManifest loaded:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")
    print(f"  Class balance: {df['label'].value_counts().to_dict()}")

    # 2. Setup Wav2Vec 2.0 Model & Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        label2id={"real": 0, "fake": 1},
        id2label={0: "real", 1: "fake"}
    )

    # Freeze feature encoder to strictly adhere to RTX 3050 6GB VRAM guardrail
    model.freeze_feature_encoder()
    print("[OK] Feature encoder frozen (model.freeze_feature_encoder() called successfully).")

    train_ds = AudioDataset(train_df, feature_extractor, max_duration=args.max_duration)
    val_ds = AudioDataset(val_df, feature_extractor, max_duration=args.max_duration)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=feature_extractor
    )

    print("\n--- Starting Fine-Tuning ---")
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"\n[OK] Fine-tuned multilingual model saved to {args.output_dir}")

    # Evaluate broken down by language
    metrics = evaluate_multilingual(model, feature_extractor, val_df, device, max_duration=args.max_duration)

if __name__ == "__main__":
    main()
