# Progressive VideoRAE

首版实现面向 `16 × 480 × 768 @ 12 FPS` 视频，使用全分辨率 V-JEPA2 特征、固定的 64 组 progressive state、帧内低频到高频因果 projector，以及可显式控制 feature cache 的 Wan2.2 decoder。

核心张量约定：

```text
RGB / target       [B, 3, 16, 480, 768]
VFM grid           [B, 8, 30, 48, C]
Progressive state  [B, 16, 30, 48, 48]
Flat state         [B, 16, 1440, 48]
Reconstruction     [B, 3, 16, 480, 768]
```

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
python -m progressive_video_rae.data.build_manifest \
  --csv-spec /share/project/liujingyi/progressive_video_rae/data/data_csv.md \
  --output-dir /share/project/liujingyi/progressive_video_rae/data/manifests \
  --split-seed 20260807
```

默认会用 PyAV 探测视频元数据；只做快速结构构建时可增加 `--no-probe-video`。输出包括总 manifest、train/val/test、固定的 val/test 1:1 balanced 子集以及统计报告。环境音和音乐按绝对路径合并，重合样本保留双 source tag；caption 始终保留。

训练入口：

```bash
torchrun --standalone --nproc_per_node=8 -m progressive_video_rae.train \
  --config configs/train/stage1a.yaml
```

推荐阶段顺序：

```bash
# Stage 1A：10k full-state warm-up
torchrun --standalone --nproc_per_node=8 -m progressive_video_rae.train \
  --config configs/train/stage1a.yaml

# Stage 1B：从 1A 权重初始化，重新开始本阶段的 90k step 计数
torchrun --standalone --nproc_per_node=8 -m progressive_video_rae.train \
  --config configs/train/stage1b.yaml \
  --init-from outputs/stage1a/step_00010000.pt

# Stage 2A：冻结 encoder/projector，训练 decoder 50k
torchrun --standalone --nproc_per_node=8 -m progressive_video_rae.train \
  --config configs/train/stage2a.yaml \
  --init-from outputs/stage1b/step_00090000.pt
```

`--resume` 用于同一阶段中断续训，会恢复 optimizer、scheduler、RNG 和 sampler epoch；跨阶段应使用 `--init-from`，只继承模型与 discriminator 权重。

评估入口：

```bash
python -m progressive_video_rae.evaluate \
  --config configs/eval/full_480p.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir /path/to/eval_output
```

默认配置不会把 V-JEPA2 输入降采样到 `240×384`。上游源码与权重位置、许可证和固定 commit 见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 固定 state 与 cache API

`CausalFrequencyProjector` 将 `30×48` 网格固定分成 64 个 set。生产 layout checksum 为：

```text
80e0ee012681f5e45a95584c9f08f17a5d1a27e753ecda54d3e91bb97384256b
```

同一 set 双向可见，query set 只能访问不晚于自己的 key set。prefix 之外的位置使用同一个可学习 mask token。Wan decoder cache 必须显式传递：

```python
first = decoder.decode(state_chunk_0, cache_mode="reset")
second = decoder.decode(
    state_chunk_1,
    cache_mode="reuse",
    cache_state=first.cache_state,
)
```

Stage 1A/1B/2A 都使用 `cache_mode="disabled"`。`reset/reuse` 仅为后续 Stage 2B 提供顺序解码接口。

## 测试

```bash
pytest -q
python -m compileall -q progressive_video_rae tests
bash -n scripts/create_env.sh scripts/download_weights.sh scripts/sync_upstream_sources.sh
```

测试覆盖固定 layout、未来 set 隔离、prefix mask、1:1 分布式 sampler、CSV 去重与 split、显式 cache 生命周期，以及直接调用固定 commit Wan2.2 源码的 full/cached 等价性。需要完整权重和 A100 的 V-JEPA2/VideoMAEv2/full-480p smoke test 应在运行下载脚本后执行。

