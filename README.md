# Progressive VideoRAE

当前实现：candidate v3 state contract + checkpoint schema v4，覆盖 NOVA 之前的
Stage 1A、Stage 1B、Stage 2-A 和 Stage 2-B。

正式几何为 `17 RGB → 5 latent states → 17 RGB`；Stage 2-B 使用 `33 RGB → 9 latent states → 33 RGB`。V-JEPA2 通过未修改的 full-attention native prefixes 提供语义，projector 产生 48 个等长 progressive sets，Wan2.2 的完整预训练时空 decoder 负责重建。

核心张量约定：

```text
RGB / target       [B, 3, 17|33, 480, 768]
V-JEPA groups      5|9 × (1|2 tubelets) × [30,48,C]
Canonical state    [B, T=5|9, S=48, K=30, C=48]
Decoder grid view  [B, 48, T=5|9, 30, 48]
Reconstruction     [B, 3, 17|33, 480, 768]
```

一个 latent frame 是一个完整 state；一个 state 包含 48 个 set，每个 set 为 `[K=30,C=48]`。`P_s` 是零起点包含端点的视图：可见 sets 为 `0…s`，full state 是 `P_47`。

## 快速开始

所有 Python、pytest 和 torchrun 命令都先进入统一环境：

```bash
source /share/project/liujingyi/activate_conda.sh
conda activate waverae
```

只生成环境和权重命令，不会在仓库初始化时自动安装或下载：

```bash
bash scripts/create_env.sh
bash scripts/download_weights.sh --model all
```

`create_env.sh` 使用相同的环境入口。手动运行仓库命令时仍需执行：

```bash
source /share/project/liujingyi/activate_conda.sh
conda activate waverae
```

它还会调用 `scripts/sync_upstream_sources.sh`，把 V-JEPA2、VideoMAEv2 和 NOVA 固定到计划指定的 commit。Wan2.2 默认直接使用现有的 `/share/project/lgy/Wan2.2`。脚本只有在用户显式运行时才会创建环境、同步源码或下载权重。

构建 manifest：

```bash
python -m progressive_videorae.data.build_manifest \
  --csv-spec /share/project/liujingyi/progressive_video_rae/data/data_csv.md \
  --output-dir /share/project/liujingyi/progressive_video_rae/data/manifests \
  --split-seed 20260807
```

默认会用 PyAV 探测视频元数据；只做快速结构构建时可增加 `--no-probe-video`。训练 manifest 按绝对路径全局去重并合并 source tags，保留最完整 caption。训练 sampler 对全部有效视频统一随机采样；声音类别和 source tags 只用于统计，不影响概率、DDP 分片或 decode retry。

训练入口：

```bash
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage1a.yaml
```

推荐阶段顺序：

```bash
# Stage 1A：10k full-state warm-up
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage1a.yaml

# Stage 1B：从 1A 权重初始化，重新开始本阶段的 90k step 计数
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage1b.yaml \
  --init-from /share/project/liujingyi/ckpts/progressive_video_rae/training/stage1a/<run_id>/step_00010000.pt

# Stage 2A：冻结 encoder/projector，训练 decoder 50k
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage2a.yaml \
  --init-from /share/project/liujingyi/ckpts/progressive_video_rae/training/stage1b/<run_id>/step_00090000.pt

# Stage 2-B：33帧 stateful full-state + REPA，20k
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage2b.yaml \
  --init-from /share/project/liujingyi/ckpts/progressive_video_rae/training/stage2a/<run_id>/step_00050000.pt
```

fresh 和 `--init-from` 运行由 rank 0 生成并广播 UTC 微秒 `run_id`。checkpoint 写入
`/share/project/liujingyi/ckpts/progressive_video_rae/training/<stage>/<run_id>/`，日志写入
`/share/project/liujingyi/logs/waverae/progressive_video_rae/<stage>_<run_id>.train.jsonl`。
`--resume` 必须指向配置 checkpoint root 内的正式 checkpoint，并复用原 run 目录和
checkpoint 记录的日志；每个 run 只维护自己的 `latest.pt`。配置会拒绝相对路径和仓库
内部的 checkpoint/log root。


`--resume` 用于同一阶段中断续训，会恢复 optimizer、scheduler、RNG 和 sampler
epoch；跨阶段应使用 `--init-from`，只继承模型与 discriminator 权重。正式交接严格为
`1A → 1B → 2-A → 2-B`，且上游 checkpoint 必须完成本阶段
`stage_max_steps`。历史短 checkpoint 或 schema v3 仅能显式增加
`--allow-smoke-checkpoint` 作 smoke/迁移；它不会放宽 stage adjacency、shape、
StateContract、上游 commit 或 V-JEPA/Wan identity 校验。

Stage 1B 每个 optimizer update 固定执行 `4 full + 3 single-prefix + 1 paired-prefix`，使用8个microbatch完成一次梯度累积；paired-prefix的两个相邻endpoint loss取平均。REPA/GAN只属于full路径。Stage 2-A/B都只使用canonical `P_47`，不构造`SpatialPrefixView`、不采样endpoint、也不计算DCT-prefix loss；两阶段都保留phase-specific REPA loss。Stage 2冻结REPA head参数，但固定head仍参与forward，REPA梯度继续回传到bridge/adapter/Wan。

ViT-L 与 ViT-g 的完整训练 checkpoint 不能相互初始化或 `--resume`。正式训练入口不再
提供 decoder-only 的 `--init-decoder-from` 旁路；切换 encoder variant 应建立独立的
Stage 1-A run 和 representation identity。冻结的 V-JEPA2 backbone 不写入训练
checkpoint，恢复时从模型配置中的官方权重重新加载。

正式训练由 rank 0 对 V-JEPA2 与 Wan 预训练文件分块计算真实 SHA256，再把成功摘要或
错误广播给所有 rank。sidecar 缺失时原子创建；已有 sidecar 会与真实内容核对，格式
错误或摘要不一致会失败且不会覆盖。`representation_identity` 只消费已经验证的摘要。
下载脚本也会生成 sidecar；已有权重可手动执行：

```bash
python -m progressive_videorae.tools.validate_checkpoints \
  --root /share/project/liujingyi/ckpts \
  --model all \
  --write-sha256
```

评估入口：

```bash
python -m progressive_videorae.evaluate \
  --config configs/eval/full_480p.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir /path/to/eval_output
```

评估仅使用完整 `P_47` state，默认从完整 test manifest 中稳定抽取 2048 个可解码
clip，报告 RGB reconstruction、V-JEPA cosine、StyleGAN-V-compatible
reconstruction FVD 以及前 8 个样本的 temporal cache parity。正式配置要求 CUDA/bf16
和经过 SHA256 校验的 I3D TorchScript。首次运行前下载评估权重：

```bash
PVR_CKPT_ROOT=/share/project/liujingyi/ckpts \
  bash scripts/download_weights.sh --model evaluation
```

结果按样本增量写入 `progress/`。中断后只有 provenance 完全一致时才能恢复：

```bash
python -m progressive_videorae.evaluate \
  --config configs/eval/full_480p.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir /path/to/eval_output \
  --resume
```

默认配置不会把 V-JEPA2 输入降采样到 `240×384`。上游源码与权重位置、许可证和固定 commit 见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 固定 state 与 cache API

State contract 固定为：

```text
layout_checksum=00c67dda3753fe5c7f800b2f20d84a1116a9acfa07fcaf5b7281910d2048c535
prefix_indexing=zero_based_inclusive_endpoint_v1
version=native_vjepa_prefix_r4_v3
layout_version=fps_v2_h30_w48_s48_k30
decoder_mode=rae_latent_causal_bridge_wan_native_r4
```

同一 set 内双向可见，set `s` 只能读取 `0…s`。未生成 sets 使用一个全局共享、严格零初始化的 learnable mask token。Wan 使用 checkpoint-compatible `temperal_upsample=[True,True,False]`（上游 downsample flags 的逆序），时间因子仍为 4。跨调用 cache 必须显式传递并校验 sequence/contract/codec/decoder identity：

```python
first = decoder.decode(state_chunk_0, cache_mode="reset", sequence_id="sample")
second = decoder.decode(
    state_chunk_1,
    cache_mode="reuse",
    cache_state=first.cache_state,
    sequence_id="sample",
)
```

三种 decode mode 都采用相同的逐 latent 顺序：
`temporal_adapter/RAE cache → inverse normalization → pre_decoder → Wan feature cache`。
`reset/reuse` 只接受携带完整 `StateContract` 的 `ProgressiveState` 或
`SpatialPrefixView`，并要求非空 `sequence_id`；raw Tensor 仅允许用于不持久化 cache 的
可信 `disabled` 路径。`disabled` 创建并丢弃一次性 dual-cache；`reset` 创建新 cache，
要求
`image_first + video_group...`；`reuse` 只接受 `video_group`。训练期间两类
cache 都保留计算图，仅推理跨调用 continuation 在公共 decode 调用边界 detach。

随机采样的每个完整 clip 都显式定义为一个新的 codec segment，而不是源视频时间 0。
dataset 返回 `codec_sequence_id`、`is_sequence_start`、
`sequence_origin=sampled_segment` 和 `segment_start_timestamp`。Stage 2-B 每个
33 帧 sample 只 reset 一次，随后执行 `z1…z8` reuse；不跨 DataLoader batch、
epoch 或不同样本保存 cache。

## 测试

```bash
pytest -q
python -m compileall -q progressive_videorae tests
bash -n scripts/create_env.sh scripts/download_weights.sh scripts/sync_upstream_sources.sh
```

当前轻量回归结果为 `107 passed, 3 skipped`，覆盖 v3 layout、未来 set 隔离、prefix mask、source-agnostic 分布式 sampler、CSV 去重与 split、四阶段 objective、REPA 梯度所有权、显式 cache 生命周期、真实内容 SHA 校验和外置 run 路径。大型集成测试通过 `PVR_RUN_LARGE_WEIGHT_LOAD_TESTS=1 pytest -q` 启用。

本轮额外对现有 V-JEPA2/Wan 文件执行真实内容 SHA256 并二次只读复核，但未加载多 GB
权重做模型 forward。训练 smoke、480p GPU forward/backward 和真实 DDP/resume 验收仍未执行。
