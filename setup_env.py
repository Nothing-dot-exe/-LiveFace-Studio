"""Automated environment verification and model setup for LiveFace-Studio."""

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

project_root = os.path.dirname(os.path.abspath(__file__))

# Register NVIDIA CUDA DLL directories on Windows
if sys.platform == "win32":
    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _candidate_dirs = []
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            _candidate_dirs.append(_torch_lib)
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    _candidate_dirs.append(_bin_dir)
        for _d in _candidate_dirs:
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass


def run_setup() -> bool:
    print("=" * 65)
    print("      LiveFace-Studio — Automated System & Model Setup")
    print("=" * 65)
    print()

    # 1. Hardware & Accelerator Check
    print("[1/4] Checking Hardware & GPU Acceleration...")
    try:
        from modules import platform_info
        platform_info.print_banner()
        accelerator = platform_info.accelerator_label()
        print(f"      Selected Accelerator: {accelerator}")
    except Exception as e:
        print(f"      [WARN] Platform detection note: {e}")

    # 2. Check Deep Learning Runtimes
    print()
    print("[2/4] Checking Deep Learning Runtimes...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"      [OK] PyTorch CUDA is ACTIVE: {device_name}")
        else:
            print("      [INFO] PyTorch CUDA not detected; running in CPU mode.")
    except Exception as e:
        print(f"      [INFO] PyTorch: {e}")

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"      [OK] ONNX Runtime Providers: {', '.join(providers)}")
    except Exception as e:
        print(f"      [WARN] ONNX Runtime: {e}")

    # 3. Download & Verify AI Models
    print()
    print("[3/4] Verifying and Downloading Required AI Models...")
    from modules.model_downloader import ensure_model, ensure_insightface_pack

    print("      - Checking Face Swapper (inswapper_128_fp16.onnx)...")
    swapper_path = ensure_model("inswapper_128_fp16.onnx")
    if swapper_path and os.path.isfile(swapper_path):
        size_mb = os.path.getsize(swapper_path) / (1024 * 1024)
        print(f"        [OK] Face Swapper ready ({size_mb:.1f} MB)")
    else:
        print("        [ERROR] Failed to download inswapper_128_fp16.onnx!")
        return False

    print("      - Checking InsightFace Buffalo_L model pack...")
    if ensure_insightface_pack("buffalo_l"):
        print("        [OK] InsightFace Buffalo_L pack ready.")
    else:
        print("        [ERROR] Failed to download Buffalo_L pack!")
        return False

    print("      - Checking Real-Time Enhancer (GPEN-BFR-256.onnx)...")
    enhancer_path = ensure_model("GPEN-BFR-256.onnx")
    if enhancer_path and os.path.isfile(enhancer_path):
        size_mb = os.path.getsize(enhancer_path) / (1024 * 1024)
        print(f"        [OK] GPEN-BFR-256 ready ({size_mb:.1f} MB)")
    else:
        print("        [WARN] GPEN-BFR-256 download skipped/failed, app will retry when enabled.")

    # 4. Check Cameras & ffmpeg
    print()
    print("[4/4] Checking Camera & System Tools...")
    ffmpeg_exe = os.path.join(project_root, "ffmpeg.exe")
    if os.path.isfile(ffmpeg_exe):
        print("      [OK] Bundled ffmpeg.exe found.")
    else:
        print("      [INFO] Bundled ffmpeg not found; system PATH will be used.")

    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("      [OK] Default camera (index 0) detected and functional.")
            cap.release()
        else:
            print("      [INFO] Camera 0 not opened (may be in use or disconnected).")
    except Exception as e:
        print(f"      [WARN] Camera test: {e}")

    print()
    print("=" * 65)
    print("  [SUCCESS] LiveFace-Studio is fully configured and ready to run!")
    print("  Double-click 'run.bat' to launch the application.")
    print("=" * 65)
    print()
    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
