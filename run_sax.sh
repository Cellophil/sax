#!/usr/bin/env bash
set -e -o pipefail

echo "[run_sax] starting at $(date -Is)"

# Resolve HOME if not set (systemd sometimes doesn't set it)
HOME_DIR=${HOME:-/home/andreas}
CONDA_BASE_DIR=${CONDA_BASE_DIR:-/home/andreas/miniforge3}

# Activate conda environment "sax"
if [ -f "$CONDA_BASE_DIR/etc/profile.d/conda.sh" ]; then
  . "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
elif [ -f "$HOME_DIR/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME_DIR/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME_DIR/anaconda3/etc/profile.d/conda.sh"
else
  # Fallback: try to initialize conda in this shell if available
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
fi

# Ensure conda is available now
if ! command -v conda >/dev/null 2>&1; then
  echo "[run_sax] conda not found in PATH. Please edit run_sax.sh to source your conda.sh." >&2
  exit 1
fi

echo "[run_sax] activating conda env 'sax'"
conda activate sax
echo "[run_sax] python: $(command -v python)"
python -V || true

# Run the Python module
exec python -m sax.steuerung
