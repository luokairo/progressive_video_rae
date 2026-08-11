# Third-party provenance

## Python environment

运行权重校验、下载后的结构检查或任何 Python 工具前，先执行：

```bash
source /share/project/liujingyi/activate_conda.sh
conda activate waverae
```

本项目只通过薄 adapter 调用公开上游实现，不把 VideoRAE 或 SpectralAR 的论文复现代码描述为官方实现。

| Project | Commit / release | Intended reuse | License action |
|---|---|---|---|
| V-JEPA2 | `204698b45b3712590f06245fbfba32d3be539812` | ViT-L/16 builder, RoPE/non-square attention | Keep upstream notices and follow checkpoint terms |
| VideoMAEv2 | `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b` | ViT adapter and intermediate features | Keep upstream notices |
| NOVA | `63c5a724fc4e264e229a95c893184434f00c9413` | Shape/configuration reference only; no random token ordering | Keep attribution for any copied utility |
| Wan2.2 | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` | `Decoder3d`, `CausalConv3d`, resampling and cache behavior | Keep Alibaba Wan copyright/license notices |

Default local Wan source root is `/share/project/lgy/Wan2.2`; it can be overridden with `PVR_WAN_SOURCE_ROOT` or the model config. The V-JEPA2 and VideoMAEv2 source roots are likewise configurable and are not silently downloaded at runtime.

正式 schema v4 训练由 rank 0 分块计算 V-JEPA2 与 Wan checkpoint 的真实 SHA256。
缺少 `.sha256` sidecar 时会原子创建；已有 sidecar 的格式或摘要不匹配时直接失败且不
覆盖，再把成功摘要或错误广播给其他 rank。`scripts/download_weights.sh` 也会在
下载/校验后生成；对已有权重可执行：

```bash
python -m progressive_videorae.tools.validate_checkpoints \
  --root /share/project/liujingyi/ckpts \
  --model all \
  --write-sha256
```
