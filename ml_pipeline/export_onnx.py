"""
VaniShield AI — Pillar 8: Edge Optimization & Deployment Tooling
Exports fine-tuned Wav2Vec 2.0 PyTorch model to ONNX FP32,
and applies Dynamic INT8 Quantization (shrinking from ~380 MB down to ~95 MB)
for sub-250ms execution on CPU edge devices, bank branch terminals, and mobile runtimes.
"""

import os
import sys
import torch
from transformers import Wav2Vec2ForSequenceClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

CHECKPOINT_DIR = r"C:\AI\VoiceCloning\results_kaggle_all"
OUTPUT_DIR = r"C:\AI\VoiceCloning\onnx_edge_models"
FP32_ONNX_PATH = os.path.join(OUTPUT_DIR, "vanishield_wav2vec2_fp32.onnx")
INT8_ONNX_PATH = os.path.join(OUTPUT_DIR, "vanishield_wav2vec2_int8.onnx")

def main():
    print("=" * 70)
    print("VaniShield AI: Pillar 8 - ONNX Dynamic INT8 Edge Export")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading PyTorch checkpoint from: {CHECKPOINT_DIR}")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    # Dummy input: 3.5 seconds of 16kHz mono audio (56,000 samples)
    dummy_input = torch.zeros(1, 56000, dtype=torch.float32)

    print(f"\n1. Exporting PyTorch to ONNX FP32...")
    torch.onnx.export(
        model,
        dummy_input,
        FP32_ONNX_PATH,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    fp32_size_mb = os.path.getsize(FP32_ONNX_PATH) / (1024 * 1024)
    print(f"  [OK] Exported FP32 ONNX -> {FP32_ONNX_PATH} ({fp32_size_mb:.1f} MB)")

    print(f"\n2. Applying Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QInt8
    )
    int8_size_mb = os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024)
    print(f"  [SUCCESS] Quantized INT8 ONNX -> {INT8_ONNX_PATH} ({int8_size_mb:.1f} MB)")

    compression_ratio = fp32_size_mb / int8_size_mb
    print("\n" + "=" * 70)
    print("EDGE COMPRESSION SUMMARY:")
    print(f"  Original FP32 Size:   {fp32_size_mb:.1f} MB")
    print(f"  Quantized INT8 Size:  {int8_size_mb:.1f} MB (Compression: {compression_ratio:.1f}x smaller)")
    print(f"  Edge Deployment Path: {INT8_ONNX_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
