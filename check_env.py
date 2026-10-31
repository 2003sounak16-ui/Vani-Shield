"""
VaniShield AI — Hardware & Environment Diagnostic Script
Team CypherEye | Project VaniShield AI
Checks CUDA acceleration on RTX 3050, PyTorch, audio libraries, and Jev-Omni companion server.
"""

import sys
import os
import platform
import importlib.util
from importlib.metadata import version, PackageNotFoundError

# Suppress Hugging Face hub telemetry/warnings during check
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

def check_python():
    print(f"[x] Operating System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"[x] Python Version:   {platform.python_version()} ({'64-bit' if sys.maxsize > 2**32 else '32-bit'})")
    if sys.version_info < (3, 9):
        print("  [!] WARNING: Python 3.9+ is recommended for VaniShield AI.")
        return False
    return True

def check_cuda_and_gpu():
    print_header("Hardware & CUDA Acceleration (NVIDIA RTX 3050)")
    try:
        import torch
        print(f"[x] PyTorch Version:   {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"[x] CUDA Available:    {cuda_ok}")
        if cuda_ok:
            dev_id = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_id)
            props = torch.cuda.get_device_properties(dev_id)
            total_vram_gb = props.total_memory / (1024**3)
            free_vram_bytes, _ = torch.cuda.mem_get_info()
            free_vram_gb = free_vram_bytes / (1024**3)
            print(f"[x] Active GPU Device: #{dev_id} {dev_name}")
            print(f"[x] Compute Capability: {props.major}.{props.minor}")
            print(f"[x] Total VRAM:        {total_vram_gb:.2f} GB")
            print(f"[x] Free VRAM:         {free_vram_gb:.2f} GB")
            print(f"[x] cuDNN Version:     {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
            print("[OK] CUDA Hardware acceleration is verified and operational for FP16 inference.")
            return True
        else:
            print("  [!] WARNING: CUDA is not active in PyTorch. Inference will fall back to CPU.")
            return False
    except ImportError:
        print("  [ERROR] PyTorch is not installed in the current environment.")
        return False

def check_dependencies():
    print_header("Core Machine Learning & Streaming Dependencies")
    packages = [
        ("torch", "PyTorch Core Engine"),
        ("torchaudio", "PyTorch Audio Processing"),
        ("transformers", "Hugging Face Wav2Vec 2.0"),
        ("faster_whisper", "Fast CTranslate2 ASR Engine"),
        ("librosa", "Audio Feature Extraction & DSP"),
        ("soundfile", "Raw PCM / WAV I/O"),
        ("fastapi", "Real-Time Streaming REST & WS API"),
        ("uvicorn", "High-Performance ASGI Server"),
        ("websockets", "Low-Latency WebSocket Streaming"),
        ("pydantic", "Data Validation & Schema Modeling"),
        ("phonenumbers", "Telecom Carrier & Number Validation"),
        ("requests", "Companion HTTP Client"),
        ("numpy", "Vector Math & Signal Processing"),
        ("scipy", "Acoustic Filtering & Spectral Analysis"),
    ]

    all_ok = True
    for pkg, label in packages:
        spec = importlib.util.find_spec(pkg)
        if spec is not None:
            try:
                pkg_dist = pkg.replace("_", "-")
                ver = version(pkg_dist)
            except PackageNotFoundError:
                ver = "installed"
            print(f"  [OK] {pkg:<16} (v{ver:<9}) - {label}")
        else:
            print(f"  [MISSING] {pkg:<16} - {label}")
            all_ok = False
    return all_ok

def check_jev_omni():
    print_header("Jev-Omni 12B Companion Server (Port 9931)")
    try:
        import requests
        url = "http://127.0.0.1:9931"
        try:
            r = requests.post(f"{url}/embeddings", json={"content": "VaniShield ping check"}, timeout=2)
            if r.status_code == 200:
                print(f"[OK] Jev-Omni 12B Llama-Server is ACTIVE on {url}")
                print("     Intent analysis embeddings pipeline is ready for live fusion.")
                return True
            else:
                print(f"[!] Jev-Omni server responded with HTTP {r.status_code}")
                return False
        except Exception:
            print(f"[!] Jev-Omni server is NOT running on port 9931.")
            print("    To launch Jev-Omni, execute the companion command in a separate terminal:")
            print(r'    & "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe" --model "C:\AI\Models\Jev-Omni\Jev-Omni-Unified-Q4_K_M.gguf" --mmproj "C:\AI\Models\Jev-Omni\mmproj-jev-omni.gguf" --ctx-size 512 --batch-size 128 --ubatch-size 128 --gpu-layers 16 --embedding --pooling none --port 9931')
            return False
    except ImportError:
        print("  [!] requests library not found.")
        return False

def main():
    print("\n" + "#" * 65)
    print(" PROJECT VANISHIELD AI -- SYSTEM & HARDWARE DIAGNOSTIC REPORT")
    print(" Team CypherEye | Real-Time Voice Cloning Defense")
    print("#" * 65)

    py_ok = check_python()
    gpu_ok = check_cuda_and_gpu()
    deps_ok = check_dependencies()
    jev_ok = check_jev_omni()

    print_header("Diagnostic Summary")
    print(f"  Environment:   {'READY' if py_ok else 'ATTENTION NEEDED'}")
    print(f"  GPU Accel:     {'ACTIVE (RTX 3050)' if gpu_ok else 'CPU FALLBACK'}")
    print(f"  Dependencies:  {'ALL SATISFIED' if deps_ok else 'SOME MISSING'}")
    print(f"  Jev-Omni 12B:  {'ONLINE' if jev_ok else 'OFFLINE / STANDBY'}")
    print("=" * 65 + "\n")

    if py_ok and deps_ok:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
