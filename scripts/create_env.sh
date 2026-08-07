#!/usr/bin/env bash
set -euo pipefail

source /share/project/liujingyi/activate_conda.sh

PVR_ENV_NAME="${PVR_ENV_NAME:-progressive_video_rae}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${PVR_ENV_NAME}"; then
  conda create -n "${PVR_ENV_NAME}" python=3.10 -y
fi
conda activate "${PVR_ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  xformers==0.0.29.post1 \
  numpy==1.26.4 transformers==4.51.3 accelerate==1.6.0 \
  timm==1.0.15 einops decord av opencv-python-headless \
  pandas pyarrow scipy scikit-image torch-dct lpips torchmetrics matplotlib \
  hydra-core omegaconf wandb tensorboard \
  "huggingface_hub[cli]" safetensors tqdm pytest ruff
python -m pip install -e ".[train,dev]"
bash scripts/sync_upstream_sources.sh

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
