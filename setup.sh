#!/usr/bin/env bash
# Recreate the 'modular' conda env and the IsaacLab checkout this project was developed with.
#
#   bash setup.sh            # env name: modular
#   ENV_NAME=foo bash setup.sh
#
# Requires: conda, git, an NVIDIA driver supporting CUDA 12.8, and Isaac Sim EULA acceptance
# (export OMNI_KIT_ACCEPT_EULA=YES to skip the prompt on first launch).
set -euo pipefail

ENV_NAME="${ENV_NAME:-modular}"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="3c6e67bb5c7ada942a6d1884ab69338f57596f77"  # tag v2.3.0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Keep ROS / user-site packages from leaking into the install.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 1. IsaacLab sources (installed editable from _isaaclab/, which is git-ignored).
if [ ! -d _isaaclab/.git ]; then
  git clone "$ISAACLAB_REPO" _isaaclab
fi
git -C _isaaclab checkout --quiet "$ISAACLAB_COMMIT"

# 2. Reference submodules (not needed to run; kept for comparison).
git submodule update --init --recursive

# 3. Conda env with the exact pip snapshot.
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

python -m pip install --no-deps -r requirements-lock.txt
for pkg in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  python -m pip install --no-deps -e "_isaaclab/source/$pkg"
done
python -m pip install --no-deps -e source/modular

echo "Done. Activate with: conda activate $ENV_NAME"
