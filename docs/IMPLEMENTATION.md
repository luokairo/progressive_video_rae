# Implementation notes

## End-to-end data flow

1. Dataset 从全量去重 manifest 均匀采样连续 `17` 或 `33` 帧，所有帧共享 crop/flip；声音类别只作统计。
2. Frozen V-JEPA2 对 `F=1+4n` 执行 native full-attention prefixes。17 帧使用 `2/6/10/14/18`，33 帧继续到 `34`；每个 prefix 提取末端 `1/2` 个 tubelets。
3. Projector 对选层做 softmax mixing。anchor 独立投影，video group 通过 phase embedding 与 learned-query attention 融合两个 tubelets。
4. `30×48=1440` 个位置按 center-first FPS 排序，形成 `[S=48,K=30]`；4 层 block-set-causal transformer 后沿 `C=48` 做 affine LayerNorm。
5. Canonical state 始终先完整计算为 `[B,T,48,30,48]`。Stage 1B 再从完整state构造prefix view：sets `0…s`保留真实token，`s+1…47`使用一个全局共享、严格零初始化的mask token；`SpatialPrefixView`不伪装成canonical state。
6. Decoder inverse-FPS 后依次经过 zero-init latent-causal bridge、near-identity StateToWan、Wan inverse normalization、预训练 `conv2 + Decoder3d`。
7. Wan 使用 checkpoint-compatible `temperal_upsample=[True,True,False]`。它是上游 `temperal_downsample=[False,True,True]` 的逆序，加载全部 `time_conv` 并保持 `1/4` 输出几何。
8. REPA 从 Wan 第一层 temporal upsample 之前捕获一个 feature/latent；anchor head 对齐首 tubelet，双 phase head 分别对齐 video-group `f0/f1`。

## Public interfaces

- `VJEPA2Encoder.encode_prefixes(pixel_values) -> PrefixEncoderOutput`
- `CausalFrequencyProjector.forward(features) -> ProjectorOutput`
- `CausalFrequencyProjector.make_prefix_view(state, endpoint) -> SpatialPrefixView`
- `WanVideoDecoder.decode(state_or_view, cache_mode, cache_state, sequence_id) -> WanDecoderOutput`

`ProgressiveState` 只保存真实 canonical tokens。`P_s` 使用零起点包含端点语义，可见数量为 `s+1`；`P_47` 直接使用 canonical state，不执行 mask replacement。

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
- `cache_mode=disabled` 也逐 latent 使用一次性内部 cache；`reset/reuse` 允许 stateful continuation，并校验 sequence、batch、dtype/device、contract、codec 与 decoder identity。
- sample 内 bridge/Wan cache 保留计算图；无关 sample 之间 reset。仅推理跨调用 continuation 可以 detach。
- `--resume` 只允许同 stage/objective，恢复 optimizer、scheduler、RNG 和 sampler epoch；`--init-from` 用于跨阶段只继承模型和 discriminator。

## Loss routing

- Full：L1、LPIPS、temporal L1、phase-specific local/global REPA、按计划启动的 PatchGAN。
- Prefix：累计低通重建、LPIPS、band、frequency leakage、paired delta；不含 REPA/GAN。
- Stage 2-A/B 日志必须为 `objective/prefix_active=0`、`objective/repa_active=1`。
- BF16 下 DCT 在 autocast-disabled 的 float32 区域计算。

## Checkpoint identity

Schema v3 保存 stage/objective、State Contract、layout checksum、codec/decoder ID、selected V-JEPA layers、上游 commits、optimizer/scheduler、RNG 和 sampler epoch。同形状但 representation identity 不同的 state/checkpoint 不得混用。
