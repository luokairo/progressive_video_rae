# Progressive VideoRAE benchmark data

This directory contains fixed protocol configs and manifest builders. It does not download or
redistribute licensed benchmark media. All formal runs reconstruct only the complete `P_47`
state; spatial prefix sets are training views and are not benchmark outputs.

## DAVIS17-Val-PVR-17x480x768

Download the official DAVIS 2017 TrainVal 480p archive and extract it without transcoding:

```bash
mkdir -p /path/to/benchmarks/davis
cd /path/to/benchmarks/davis
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
```

The extracted root passed to `--davis-root` must contain both
`ImageSets/2017/val.txt` and `JPEGImages/480p/<sequence>/*.jpg`. Build the manifest with:

```bash
cd /share/project/liujingyi/progressive_video_rae
python -m evaluation.prepare_benchmark davis \
  --davis-root /path/to/benchmarks/davis/DAVIS \
  --output /path/to/manifests/davis17_val_480x768.parquet
```

The command requires exactly 30 unique validation sequences. Each sequence contributes one
centered clip using indices spaced by two source frames (official 24 FPS to 12 FPS), and the
evaluator reads the original JPEGs directly.

## TokenBench-PVR-17x480x768

Clone the official repository to freeze the authoritative list and preprocessing script:

```bash
cd /path/to/benchmarks
git clone https://github.com/NVlabs/TokenBench.git
cd TokenBench
python -m pip install imageio mediapy
```

Obtain each upstream dataset from its owner and accept its license/terms before downloading:

- BDD100K: <https://bdd-data.berkeley.edu/>
- BridgeData V2: <https://rail-berkeley.github.io/bridgedata/>
- Panda-70M: <https://snap-research.github.io/Panda-70M/>
- EgoExo-4D: <https://ego-exo4d-data.org/>

The NVIDIA script is `token_bench/video/preprocessing_script.py`. It has a hard-coded
`raw_video_dir = "/root/dataset"` and expects source folders named `bdd_100`, `egoexo4D`,
`panda`, and `bridgev2`. Set that variable to your licensed local root, arrange only the
officially listed samples under those folders, then run:

```bash
cd /path/to/benchmarks/TokenBench
python token_bench/video/preprocessing_script.py
```

Do not rename samples in a way that loses their `list.txt` relative identity. In particular,
EgoExo-4D and BridgeData entries have nested relative paths and repeated basenames. The official
script expects BridgeData inputs to have already been converted to MP4. Preserve the list-relative
path when doing that conversion (for example, `.../images0.mp4`) so entries remain unique. The PVR
importer resolves an exact relative video path, the same path with an `.mp4` suffix, and finally a
unique suffix/basename. Ambiguity and raw frame directories are hard errors for TokenBench; the
formal manifest always contains exactly 500 video files.

Build the exhaustive 500-sample manifest:

```bash
cd /share/project/liujingyi/progressive_video_rae
python -m evaluation.prepare_benchmark tokenbench \
  --official-list /path/to/benchmarks/TokenBench/token_bench/video/list.txt \
  --bdd100k-root /path/to/tokenbench/bdd100k \
  --bridgedata-v2-root /path/to/tokenbench/bridgedata_v2 \
  --panda-70m-root /path/to/tokenbench/panda_70m \
  --egoexo-4d-root /path/to/tokenbench/egoexo_4d \
  --output /path/to/manifests/tokenbench_480x768.parquet
```

Preparation verifies the official 100/100/100/200 source counts, all 500 paths, unique IDs,
per-source SHA256 values, official-list SHA256, and a deterministic manifest-row digest. It
writes a JSON sidecar next to the parquet. Missing or ambiguous entries are never substituted.

## Four-stage evaluation

Use the same manifest for every stage. Stage 2-B is deliberately evaluated with
`configs/model/full_480p.yaml` at 17 RGB frames / 5 temporal latents even though its training
geometry was 33 / 9.

```bash
pvr-evaluate \
  --config evaluation/configs/tokenbench_480x768.yaml \
  --benchmark-manifest /path/to/manifests/tokenbench_480x768.parquet \
  --checkpoint /path/to/stage1a/final.pt \
  --expected-stage stage1a \
  --output-dir /path/to/results/tokenbench/stage1a
```

Repeat with `stage1b`, `stage2a`, and `stage2b`; use
`evaluation/configs/davis17_val_480x768.yaml` for DAVIS. Formal stage checkpoints must be schema
v4 and have `stage_complete=true`.

After all eight runs complete:

```bash
pvr-compare-stages \
  --run-dir /results/tokenbench/stage1a \
  --run-dir /results/tokenbench/stage1b \
  --run-dir /results/tokenbench/stage2a \
  --run-dir /results/tokenbench/stage2b \
  --run-dir /results/davis/stage1a \
  --run-dir /results/davis/stage1b \
  --run-dir /results/davis/stage2a \
  --run-dir /results/davis/stage2b \
  --output-dir /results/stage_comparison
```

The comparison reports trajectory deltas only. Without a separate full-only control training
run, it must not be presented as proving that every difference was caused solely by
coarse-to-fine supervision.
