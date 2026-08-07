# Implementation notes

## Model data flow

1. Dataset 解码同一个 `16×480×768` crop，返回 `[0,1]` RGB 和 caption/路径/来源元数据。
2. V-JEPA2 adapter 使用 ImageNet mean/std、RoPE、SDPA 和非正方形网格，融合第 8/12/16/20/24 层。
3. Temporal ConvTranspose3D 将 `[8,30,48]` 恢复为 `[16,30,48]`，空间不插值。
4. 每帧 projector 使用固定 set causal mask，生成 normalized 48-channel Wan state。
5. prefix 训练把未激活 set 替换为共享 mask token；RGB 目标使用相应矩形 DCT prefix。
6. Wan2.2 wrapper 加载 `conv2 + Decoder3d`，关闭所有 temporal upsample，再用上游 unpatchify 完成 16×空间恢复。

## Public interfaces

- `VideoFoundationEncoder.forward(pixel_values, output_layers) -> EncoderOutput`
- `CausalFrequencyProjector.forward(features, prefix_len) -> ProgressiveState`
- `WanVideoDecoder.decode(state, prefix_len, cache_mode, cache_state) -> WanDecoderOutput`
- `ProgressiveVideoRAE.forward(pixel_values, prefix_len, cache_mode) -> ProgressiveVideoRAEOutput`

`ProgressiveState.metadata["unmasked_tokens"]` 只用于损失和诊断；decoder 始终读取已应用 prefix mask 的 `tokens`。

## Weight loading

- V-JEPA2 清理 `module.`、`backbone.` 和 `encoder.` 前缀后加载官方 encoder state。
- VideoMAEv2 支持 `model`/`module` checkpoint 容器并忽略分类 head。
- Wan2.2 只加载 `conv2.*` 和 `decoder.*`。由于 temporal upsampling 被关闭，checkpoint 中对应的 `time_conv` keys 被明确记录为预期忽略项，其他 unexpected key 会报错。

## Runtime boundaries

- V-JEPA2 冻结并在 `torch.no_grad()` 下执行；projector/decoder 根据 stage 切换 trainable 状态。
- Full-state loss：L1、LPIPS、REPA local/global、0.1 PatchGAN。
- Prefix loss：L1、0.5 LPIPS，不使用 REPA/GAN。
- REPA 从 Wan 第一个 upsample block 获取 `[T,60,96]` 特征，池化到 `[8,30,48]` 后投影到 VFM channel。
- BF16 下 DCT 强制在 FP32 autocast-disabled 区域计算。
- Feature cache 没有模块级跨 batch 隐式状态；`reset` 创建新 state，`reuse` 必须接收上次返回的 state。

