# Progressive VideoRAE

当前实现：candidate v3，覆盖 NOVA 之前的 Stage 1A、Stage 1B、Stage 2-A 和 Stage 2-B。

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

只生成环境和权重命令，不会在仓库初始化时自动安装或下载：

```bash
bash scripts/create_env.sh
bash scripts/download_weights.sh --model all
```

`create_env.sh` 的第一条环境命令固定为：

```bash
source /share/project/liujingyi/activate_conda.sh
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
  --init-from outputs/stage1a/step_00010000.pt

# Stage 2A：冻结 encoder/projector，训练 decoder 50k
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage2a.yaml \
  --init-from outputs/stage1b/step_00090000.pt

# Stage 2-B：33帧 stateful full-state + REPA，20k
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage2b.yaml \
  --init-from outputs/stage2a/step_00050000.pt
```

`--resume` 用于同一阶段中断续训，会恢复 optimizer、scheduler、RNG 和 sampler epoch；跨阶段应使用 `--init-from`，只继承模型与 discriminator 权重。

Stage 1B 每个 optimizer update 固定执行 `4 full + 3 single-prefix + 1 paired-prefix`，使用8个microbatch完成一次梯度累积；paired-prefix的两个相邻endpoint loss取平均。REPA/GAN只属于full路径。Stage 2-A/B都只使用canonical `P_47`，不构造`SpatialPrefixView`、不采样endpoint、也不计算DCT-prefix loss；两阶段都保留phase-specific REPA loss。Stage 2冻结REPA head参数，但固定head仍参与forward，REPA梯度继续回传到bridge/adapter/Wan。

切换到 ViT-g 时覆盖模型配置，并只复用 ViT-L 阶段训练好的 Wan decoder：

```bash
torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train \
  --config configs/train/stage1a.yaml \
  --model-config configs/model/full_480p_vitg.yaml \
  --init-decoder-from outputs/vitl_stage1b/step_00090000.pt
```

ViT-L 与 ViT-g 的完整训练 checkpoint 不能相互 `--resume`；冻结的 V-JEPA2 backbone 不写入训练 checkpoint，恢复时从模型配置中的官方权重重新加载。

评估入口：

```bash
python -m progressive_videorae.evaluate \
  --config configs/eval/full_480p.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir /path/to/eval_output
```

默认配置不会把 V-JEPA2 输入降采样到 `240×384`。上游源码与权重位置、许可证和固定 commit 见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 固定 state 与 cache API

State contract 固定为：

```text
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

训练默认启用cache-aware gradient checkpointing。Stage 1A/1B/2-A 的 `disabled` 模式内部同样逐 latent 使用临时 cache；Stage 2-B 从 `z0` reset 并对 `z1…z8` stateful reuse。

## 测试

```bash
pytest -q
python -m compileall -q progressive_videorae tests
bash -n scripts/create_env.sh scripts/download_weights.sh scripts/sync_upstream_sources.sh
```

测试覆盖 v3 layout、未来 set 隔离、prefix mask、source-agnostic 分布式 sampler、CSV 去重与 split、四阶段 objective、REPA 梯度所有权和显式 cache 生命周期。大型集成测试通过 `PVR_RUN_LARGE_WEIGHT_LOAD_TESTS=1 pytest -q` 启用。
