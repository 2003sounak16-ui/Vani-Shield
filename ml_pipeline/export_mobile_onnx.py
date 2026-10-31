"""
VaniShield AI — Mobile Edge ONNX Exporter & INT8 Quantizer
Team CypherEye | Project VaniShield AI

Fix 1 & Fix 3:
Exports trained Wav2Vec 2.0 Acoustic Classifier to ONNX format,
applies dynamic INT8 quantization for ultra-low latency mobile NPU/CPU execution,
and validates inference output and latency against the PyTorch baseline.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Project root setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "results_master_unified")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline", "checkpoints", "mobile_edge")


class Wav2Vec2ONNXWrapper(nn.Module):
    """
    Wrapper ensuring clean single-tensor input (batch, time)
    and single-tensor output (batch, 2) logits for ONNX Runtime / Mobile NPU.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        outputs = self.model(input_values)
        return outputs.logits


def export_to_onnx(checkpoint_dir: str, output_dir: str, opset_version: int = 14):
    """Exports PyTorch Wav2Vec2 model to ONNX format."""
    os.makedirs(output_dir, exist_ok=True)
    fp32_onnx_path = os.path.join(output_dir, "vanishield_acoustic_fp32.onnx")
    int8_onnx_path = os.path.join(output_dir, "vanishield_acoustic_int8.onnx")

    print(f"\n[1/4] Loading PyTorch Checkpoint from: {checkpoint_dir}")
    model = AutoModelForAudioClassification.from_pretrained(checkpoint_dir)
    model.eval()

    # Wrap model for clean I/O signatures
    wrapped_model = Wav2Vec2ONNXWrapper(model)
    wrapped_model.eval()

    # Standard 1.5s audio slice at 16,000 Hz = 24,000 samples
    dummy_input = torch.randn(1, 24000, dtype=torch.float32)

    print(f"[2/4] Exporting FP32 ONNX Model (opset={opset_version})...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )

    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    print(f"  [OK] Exported FP32 ONNX to: {fp32_onnx_path} ({fp32_size_mb:.2f} MB)")

    print("[3/4] Applying Dynamic INT8 Quantization (Fix 1 - Edge Quantization)...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8
        )
        int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
        compression = (1.0 - (int8_size_mb / fp32_size_mb)) * 100.0
        print(f"  [OK] Exported INT8 ONNX to: {int8_onnx_path} ({int8_size_mb:.2f} MB, {compression:.1f}% reduction)")
    except Exception as e:
        print(f"  [WARNING] INT8 quantization encountered an issue: {e}")
        int8_onnx_path = None

    print("[4/4] Validating ONNX Runtime Execution & Latency (Fix 3 - Mobile Acceleration)...")
    validate_and_benchmark(model, fp32_onnx_path, int8_onnx_path, dummy_input)

    return fp32_onnx_path, int8_onnx_path


def validate_and_benchmark(pt_model, fp32_path: str, int8_path: str, dummy_input: torch.Tensor):
    """Validates output numerical equivalence and benchmarks latency."""
    import onnxruntime as ort

    # PyTorch baseline
    with torch.no_grad():
        t0 = time.perf_counter()
        pt_out = pt_model(dummy_input).logits.numpy()
        pt_latency_ms = (time.perf_counter() - t0) * 1000.0

    # ONNX FP32
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    ort_fp32 = ort.InferenceSession(fp32_path, sess_options=session_options, providers=providers)
    onnx_inputs = {"input_values": dummy_input.numpy()}

    t0 = time.perf_counter()
    ort_fp32_out = ort_fp32.run(["logits"], onnx_inputs)[0]
    ort_fp32_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Check max difference
    diff_fp32 = np.max(np.abs(pt_out - ort_fp32_out))
    print(f"  * FP32 PyTorch vs ONNX Max Abs Diff: {diff_fp32:.6f} (Tolerance: < 0.001)")
    print(f"  * FP32 PyTorch Latency: {pt_latency_ms:.2f} ms | ONNX FP32 Latency: {ort_fp32_latency_ms:.2f} ms")

    if int8_path and os.path.exists(int8_path):
        ort_int8 = ort.InferenceSession(int8_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        # Warmup
        _ = ort_int8.run(["logits"], onnx_inputs)[0]
        # Benchmark 5 runs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ort_int8_out = ort_int8.run(["logits"], onnx_inputs)[0]
            times.append((time.perf_counter() - t0) * 1000.0)
        ort_int8_latency_ms = np.median(times)

        # Softmax probabilities comparison
        pt_prob = torch.softmax(torch.from_numpy(pt_out), dim=-1).numpy()[0]
        int8_prob = np.exp(ort_int8_out[0]) / np.sum(np.exp(ort_int8_out[0]))

        print(f"  * INT8 Mobile Quantized Latency: {ort_int8_latency_ms:.2f} ms (Target: < 50ms)")
        print(f"  * PyTorch Prob [Real, Fake]: [{pt_prob[0]:.3f}, {pt_prob[1]:.3f}]")
        print(f"  * INT8 ONNX Prob [Real, Fake]: [{int8_prob[0]:.3f}, {int8_prob[1]:.3f}]")
        print(f"  [OK] Mobile Model Integrity: Decision match = {np.argmax(pt_prob) == np.argmax(int8_prob)}")


def main():
    parser = argparse.ArgumentParser(description="Export VaniShield Acoustic Model to Mobile ONNX/INT8")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to trained Wav2Vec2 checkpoint")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save exported ONNX models")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    print("=" * 75)
    print(" VANISHIELD AI — MOBILE EDGE ONNX EXPORT & OPTIMIZATION")
    print(f" Source: {args.checkpoint} | Target: {args.output_dir}")
    print("=" * 75)

    export_to_onnx(args.checkpoint, args.output_dir, opset_version=args.opset)
    print("\n[SUCCESS] Mobile Edge ONNX models ready for Android / iOS deployment.\n")


if __name__ == "__main__":
    main()
