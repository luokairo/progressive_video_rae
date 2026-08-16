#!/usr/bin/env bash
set -euo pipefail

source /share/project/liujingyi/activate_conda.sh
conda activate waverae
cd /share/project/liujingyi/progressive_video_rae

exec python scripts/stage1a_ablation.py launch "$@"
