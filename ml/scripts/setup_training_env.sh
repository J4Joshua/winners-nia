#!/usr/bin/env bash
# Prepare Python deps for Song Tower training (PyTorch + HF stack).
#
# Usage (from repo root):
#   ./ml/scripts/setup_training_env.sh              # auto: extras if torch exists, else full
#   ./ml/scripts/setup_training_env.sh --extras-only   # RunPod / image already has PyTorch
#   ./ml/scripts/setup_training_env.sh --full          # fresh env: pip installs torch too
#
# Environment:
#   HF_TOKEN          optional; needed for `ml train --push-to-hub` / dataset auth if gated
#   PIP_EXTRA_ARGS    optional; e.g. "--index-url https://download.pytorch.org/whl/cu124"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
REQ_FULL="${REPO_ROOT}/ml/requirements.txt"
REQ_EXTRAS="${REPO_ROOT}/ml/requirements-train-extras.txt"

die() { echo "error: $*" >&2; exit 1; }

has_torch() {
  "${PYTHON}" -c "import torch" 2>/dev/null
}

torch_version() {
  "${PYTHON}" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown"
}

extras_only=false
full_install=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras-only|--runpod) extras_only=true ;;
    --full) full_install=true ;;
    -h|--help)
      grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
  shift || true
done

echo "Repo: ${REPO_ROOT}"
echo "Python: $(${PYTHON} --version 2>&1)"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  die "${PYTHON} not found"
fi

${PYTHON} -m pip install --upgrade pip ${PIP_EXTRA_ARGS:-}

if ${extras_only}; then
  echo "Installing training extras only (no torch from pip) → ${REQ_EXTRAS}"
  ${PYTHON} -m pip install ${PIP_EXTRA_ARGS:-} -r "${REQ_EXTRAS}"
  has_torch || die "PyTorch not found; use a PyTorch base image or run with --full"
elif ${full_install}; then
  echo "Full install (includes PyTorch from pip) → ${REQ_FULL}"
  ${PYTHON} -m pip install ${PIP_EXTRA_ARGS:-} -r "${REQ_FULL}"
else
  if has_torch; then
    echo "PyTorch already present ($(torch_version)); installing extras only."
    ${PYTHON} -m pip install ${PIP_EXTRA_ARGS:-} -r "${REQ_EXTRAS}"
  else
    echo "PyTorch not found; installing full stack from ${REQ_FULL}"
    ${PYTHON} -m pip install ${PIP_EXTRA_ARGS:-} -r "${REQ_FULL}"
  fi
fi

echo ""
if has_torch; then
  "${PYTHON}" <<'PY'
import sys
try:
    import torch
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda: {torch.version.cuda}  device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("torch check failed:", e)
    sys.exit(1)
PY
fi

echo ""
echo "Train when ready:"
echo "  cd ${REPO_ROOT}"
echo "  python -m ml train --gpu-preset h100 --output-dir ml/export"
echo ""
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Tip: export HF_TOKEN for Hugging Face Hub uploads and reliable dataset access."
fi
