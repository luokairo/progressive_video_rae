#!/usr/bin/env bash
set -euo pipefail

PVR_CKPT_ROOT="${PVR_CKPT_ROOT:-/share/project/liujingyi/ckpts}"
PVR_MODEL="all"
PVR_VERIFY_ONLY=0
PVR_I3D_URL="${PVR_I3D_URL:-https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) PVR_MODEL="$2"; shift 2 ;;
    --verify-only) PVR_VERIFY_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--model vjepa2|wan2.2|videomaev2|evaluation|all] [--verify-only]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${PVR_MODEL}" in
  vjepa2|wan2.2|videomaev2|evaluation|all) ;;
  *) echo "Unsupported model: ${PVR_MODEL}" >&2; exit 2 ;;
esac

mkdir -p \
  "${PVR_CKPT_ROOT}/vjepa2/vitl" \
  "${PVR_CKPT_ROOT}/wan2.2/ti2v_5b" \
  "${PVR_CKPT_ROOT}/videomaev2/vitb" \
  "${PVR_CKPT_ROOT}/evaluation"

download_vjepa2() {
  wget -c https://dl.fbaipublicfiles.com/vjepa2/vitl.pt \
    -O "${PVR_CKPT_ROOT}/vjepa2/vitl/vitl.pt"
}

download_wan() {
  command -v hf >/dev/null || { echo "Install huggingface_hub[cli] first" >&2; exit 1; }
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir "${PVR_CKPT_ROOT}/wan2.2/ti2v_5b"
}

download_videomae2() {
  command -v hf >/dev/null || { echo "Install huggingface_hub[cli] first" >&2; exit 1; }
  hf download OpenGVLab/VideoMAE2 distill/vit_b_k710_dl_from_giant.pth \
    --local-dir "${PVR_CKPT_ROOT}/videomaev2/vitb"
}

download_evaluation() {
  wget -c "${PVR_I3D_URL}" \
    -O "${PVR_CKPT_ROOT}/evaluation/i3d_torchscript.pt"
}

if [[ "${PVR_VERIFY_ONLY}" -eq 0 ]]; then
  [[ "${PVR_MODEL}" == "vjepa2" || "${PVR_MODEL}" == "all" ]] && download_vjepa2
  [[ "${PVR_MODEL}" == "wan2.2" || "${PVR_MODEL}" == "all" ]] && download_wan
  [[ "${PVR_MODEL}" == "videomaev2" || "${PVR_MODEL}" == "all" ]] && download_videomae2
  [[ "${PVR_MODEL}" == "evaluation" || "${PVR_MODEL}" == "all" ]] && download_evaluation
fi

python -m progressive_video_rae.tools.validate_checkpoints \
  --root "${PVR_CKPT_ROOT}" --model "${PVR_MODEL}" --write-sha256

