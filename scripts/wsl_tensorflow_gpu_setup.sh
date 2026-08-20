#!/usr/bin/env bash
# Run inside WSL2 Ubuntu from the crypto_horserace project root.
# Windows: NVIDIA driver already installed; verify with: wsl nvidia-smi
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found in WSL. Update the NVIDIA driver on Windows, reboot, then run: wsl nvidia-smi"
  exit 1
fi

echo "=== GPU from WSL ==="
nvidia-smi

VENV="${WSL_VENV_PATH:-$ROOT/.venv-wsl}"
echo "=== Creating venv: $VENV ==="
set +e
python3 -m venv "$VENV"
venv_rc=$?
set -e
if [ "$venv_rc" -ne 0 ]; then
  VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '3')"
  echo ""
  echo "ERROR: python3 -m venv failed (common on fresh Ubuntu WSL: ensurepip missing)."
  echo "Install venv support, then delete the broken folder and re-run this script:"
  echo "  sudo apt update"
  echo "  sudo apt install -y python3-venv python3-pip build-essential"
  echo "  # if apt still complains, use your exact Python version:"
  echo "  sudo apt install -y python${VER}-venv"
  echo "  rm -rf \"$VENV\""
  echo "  ./scripts/wsl_tensorflow_gpu_setup.sh"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

python -m pip install -U pip wheel setuptools

echo "=== TensorFlow + bundled CUDA/cuDNN (Linux pip) ==="
pip install "tensorflow[and-cuda]>=2.16.0"

echo "=== Other dependencies (no duplicate tensorflow line) ==="
pip install -r requirements-gpu-wsl.txt

echo "=== PyTorch (CPU-only; avoids CUDA 13 wheels clashing with TensorFlow CUDA 12) ==="
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cpu

echo "=== Verify GPU ==="
python <<'PY'
import tensorflow as tf
print("TF", tf.__version__)
print("GPUs", tf.config.list_physical_devices("GPU"))
PY

echo ""
echo "Done. Activate later with:"
echo "  source $VENV/bin/activate"
echo "Run project from this directory:"
echo "  python main.py"
