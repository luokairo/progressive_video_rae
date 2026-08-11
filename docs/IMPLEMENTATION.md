# Implementation notes

## Python environment

运行本文中的 Python、pytest 或 torchrun 命令前，统一执行：

```bash
source /share/project/liujingyi/activate_conda.sh
conda activate waverae
```

## End-to-end data flow

1. Dataset 从全量去重 manifest 均匀采样连续 `17` 或 `33` 帧，所有帧共享
   crop/flip；声音类别只作统计。每个随机 clip 显式成为新的 codec segment，并返回
   `codec_sequence_id`、`is_sequence_start`、
   `sequence_origin=sampled_segment` 与 `segment_start_timestamp`；这不表示源视频
   时间 0。
2. Frozen V-JEPA2 对 `F=1+4n` 执行 native full-attention prefixes。17 帧使用 `2/6/10/14/18`，33 帧继续到 `34`；每个 prefix 提取末端 `1/2` 个 tubelets。
3. Projector 对选层做 softmax mixing。anchor 独立投影，video group 通过 phase embedding 与 learned-query attention 融合两个 tubelets。
4. `30×48=1440` 个位置按 center-first FPS 排序，形成 `[S=48,K=30]`；4 层 block-set-causal transformer 后沿 `C=48` 做 affine LayerNorm。
5. Canonical state 始终先完整计算为 `[B,T,48,30,48]`。Stage 1B 再从完整state构造prefix view：sets `0…s`保留真实token，`s+1…47`使用一个全局共享、严格零初始化的mask token；`SpatialPrefixView`不伪装成canonical state。
6. Decoder inverse-FPS 后对每个 latent 依次执行 zero-init latent-causal bridge、
   near-identity StateToWan、Wan inverse normalization、预训练
   `conv2 + Decoder3d`，并立即更新 RAE history 与 Wan feature cache。
7. Wan 使用 checkpoint-compatible `temperal_upsample=[True,True,False]`。它是上游 `temperal_downsample=[False,True,True]` 的逆序，加载全部 `time_conv` 并保持 `1/4` 输出几何。
8. REPA 从 Wan 第一层 temporal upsample 之前捕获一个 feature/latent；anchor head 对齐首 tubelet，双 phase head 分别对齐 video-group `f0/f1`。

## Public interfaces

- `VJEPA2Encoder.encode_prefixes(pixel_values) -> PrefixEncoderOutput`
- `CausalFrequencyProjector.forward(features) -> ProjectorOutput`
- `CausalFrequencyProjector.make_prefix_view(state, endpoint) -> SpatialPrefixView`
- `WanVideoDecoder.decode(state_or_view, cache_mode, cache_state, sequence_id) -> WanDecoderOutput`

`ProgressiveState` 只保存真实 canonical tokens。`P_s` 使用零起点包含端点语义，可见数量为 `s+1`；`P_47` 直接使用 canonical state，不执行 mask replacement。
StateContract 同时固定 `prefix_indexing=zero_based_inclusive_endpoint_v1` 和 candidate
layout checksum `00c67dda3753fe5c7f800b2f20d84a1116a9acfa07fcaf5b7281910d2048c535`。
state/view/decoder 每次调用都交叉校验这两个 layout 身份字段。


## Training objectives

| Stage | RGB | Objective | Prefix | REPA |
|---|---:|---|---|---|
| Stage 1A | 17 | `full_repa` | 无 | 训练 head，loss 开启 |
| Stage 1B | 17 | `full_repa_spatial_prefix` | 仅 prefix microbatches | 仅 full microbatches |
| Stage 2-A | 17 | `full_repa` | 无 | head 冻结，loss 开启 |
| Stage 2-B | 33 | `full_repa_stateful` | 无 | head 冻结，loss 开启 |

Stage 2-A/B 启动校验拒绝任何 prefix 配置。冻结 REPA head 不会切断 autograd；固定 head 的输入梯度继续更新 bridge、StateToWan 和 Wan decoder。

Stage 1B 每个optimizer update固定执行4个full、3个single-prefix和1个paired-prefix microbatch。single endpoint为`0…46`，paired endpoint为`1…47`；任务在单步内部打乱，paired的两个相邻endpoint顺序解码并对loss取平均。DCT是逐RGB帧float32 orthonormal 2D DCT，不做时间DCT。

Wan整体保持cache-aware gradient checkpointing，以满足480p训练的显存约束。

## Weight loading and runtime boundaries

- V-JEPA2 commit：`204698b45b3712590f06245fbfba32d3be539812`，永久 `eval/no_grad/frozen`，不写入训练 checkpoint。
- Wan2.2 commit：`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`。
- `conv2 + Decoder3d` coverage 必须至少 99%；非白名单 missing、unexpected 或 shape mismatch 阻止训练。
- 训练默认启用cache-aware non-reentrant gradient checkpointing；每个latent反向重算从该调用前的feature-cache snapshot开始，不污染正式continuation cache。
- `cache_mode=disabled` 也逐 latent 使用一次性内部 cache，并保留 raw Tensor 可信入口；
  `reset/reuse` 仅接受 `ProgressiveState`/`SpatialPrefixView`，强制非空 sequence ID，
  并校验 sequence、batch、dtype/device、完整 StateContract、codec 与 decoder identity。
- sample 内 bridge/Wan cache 保留计算图；无关 sample 之间 reset。仅推理跨调用 continuation 可以 detach。
- `--resume` 只允许同 stage/objective，恢复 optimizer、scheduler、RNG 和 sampler epoch；`--init-from` 用于跨阶段只继承模型和 discriminator。
- `disabled`、`reset`、`reuse` 共用逐 latent 实现。`disabled` 使用一次性
  dual-cache；`reset` 要求 `image_first + video_group...`；`reuse` 只允许
  `video_group`。Stage 2-B 每个 33 帧 sampled segment 只 reset 一次，不跨 batch、
  epoch 或样本保存 cache。

训练 bundle 在所有 CLI override 之后执行几何 preflight：

- Stage 1A/1B/2-A 必须为 17 RGB 帧与 5 latents；
- Stage 2-B 必须为 33 RGB 帧与 9 latents，最大 V-JEPA prefix 为 34 且不超过
  context 64；
- 所有阶段满足 `F=1+4n`，model/data/encoder/decoder 空间尺寸一致；
- Stage 2 配置中禁止 prefix、DCT 与 mask-replacement 选项；
- checkpoint/log root 必须是仓库外绝对路径。fresh/init 使用 rank 0 广播的 UTC 微秒
  `run_id`；resume 复用 checkpoint 所在 run 目录和已记录日志。每个 run 只维护本目录
  的 `latest.pt`。

## Loss routing

- Full：L1、LPIPS、temporal L1、phase-specific local/global REPA、按计划启动的 PatchGAN。
- Prefix：累计低通重建、LPIPS、band、frequency leakage、paired delta；不含 REPA/GAN。
- Stage 2-A/B 日志必须为 `objective/prefix_active=0`、`objective/repa_active=1`。
- BF16 下 DCT 在 autocast-disabled 的 float32 区域计算。
- `prefix_objective_weight` 对 single-prefix total 乘一次；paired-prefix 先平均相邻
  两个 endpoint total，再乘一次。
- REPA 额外记录 anchor、每个 group 的 `f0/f1` error，以及 Stage 2-B tail-four
  mean/worst；这些统计不改变 total loss。
- `GlobalMetricWindow` 分开累计 single/paired endpoint histogram，并按 endpoint
  汇总 L1、band、leakage 与 paired-delta，在日志窗口末统一做一次 DDP reduce。

## Checkpoint identity

Schema v4 保存 `stage_max_steps`、`stage_complete`、stage/objective、StateContract、
layout version/checksum、codec/decoder ID、V-JEPA variant/selected layers、上游 commits、
optimizer/scheduler、RNG、sampler epoch、run ID、checkpoint/log 路径和来源 checkpoint。
rank 0 分块计算 V-JEPA/Wan 文件真实 SHA256：缺失 sidecar 时原子创建，已有 sidecar
必须与内容匹配；成功摘要或错误广播给所有 rank。representation identity 不直接信任
sidecar 文本。projector 与 bridge hash 按 tensor 名称、shape、dtype 和连续字节确定性计算。

加载顺序为：校验 schema/阶段/contract/静态 identity，校验 checkpoint 内 learned
hash，加载模型，再重新计算 projector/bridge hash。正式阶段矩阵固定为
`1A fresh → 1B ← 1A → 2-A ← 1B → 2-B ← 2-A`；`--resume` 只能恢复同
stage/objective。短 checkpoint 或 schema v3 只有显式
`--allow-smoke-checkpoint` 才可用于 smoke，且不能绕过 adjacency、shape、
StateContract、upstream commit 或静态 identity 校验。schema v3 smoke 迁移只补固定
prefix indexing，并要求旧 representation identity 提供完全匹配的 layout checksum；
schema v4 缺少新字段直接拒绝。

本轮不实现 Wan weight drift、固定官方 latent decode drift、PSNR/LPIPS 健康监控。
训练日志继续保留各参数组实际 LR 与 gradient norm，目标是优先保证 VideoRAE decoder
可训练以及 checkpoint 可安全交接。
