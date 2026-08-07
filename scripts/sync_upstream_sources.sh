#!/usr/bin/env bash
set -euo pipefail

PVR_UPSTREAM_ROOT="${PVR_UPSTREAM_ROOT:-/share/project/liujingyi/progressive_video_rae/third_party/upstream}"
mkdir -p "${PVR_UPSTREAM_ROOT}"

sync_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="${PVR_UPSTREAM_ROOT}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

sync_repo vjepa2 https://github.com/facebookresearch/vjepa2.git \
  204698b45b3712590f06245fbfba32d3be539812
sync_repo videomaev2 https://github.com/OpenGVLab/VideoMAEv2.git \
  29eab1e8a588d1b3ec0cdec7b03a86cca491b74b
sync_repo nova https://github.com/baaivision/NOVA.git \
  63c5a724fc4e264e229a95c893184434f00c9413

