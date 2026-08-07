from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from progressive_video_rae.model.pretrained import load_validated_pretrained
from progressive_video_rae.model.encoders.vjepa2 import VJEPA2Encoder
from progressive_video_rae.model.encoders.videomaev2 import VideoMAEv2Encoder
from progressive_video_rae.model.wan_decoder import WanVideoDecoder


class TinyBackbone(nn.Module):
    def __init__(self, depth: int = 3):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(depth)])


def test_validated_loader_rejects_empty_and_partial_checkpoints(tmp_path: Path):
    module = TinyBackbone()
    checkpoint = {key: value.detach().clone() for key, value in module.state_dict().items()}
    report = load_validated_pretrained(
        module,
        checkpoint,
        component="toy",
        checkpoint_path=tmp_path / "toy.pt",
        required_groups={"patch": ("patch_embed.*",), "last": ("blocks.2.*",)},
        minimum_coverage=1.0,
    )
    assert report.ready
    assert report.coverage == 1.0

    with pytest.raises(RuntimeError, match="no compatible tensors"):
        load_validated_pretrained(
            TinyBackbone(),
            {},
            component="toy",
            checkpoint_path=tmp_path / "empty.pt",
            required_groups={"patch": ("patch_embed.*",)},
        )

    partial = dict(checkpoint)
    partial.pop("blocks.2.weight")
    with pytest.raises(RuntimeError, match="missing model keys|missing critical groups"):
        load_validated_pretrained(
            TinyBackbone(),
            partial,
            component="toy",
            checkpoint_path=tmp_path / "partial.pt",
            required_groups={"last": ("blocks.2.*",)},
            minimum_coverage=0.95,
        )


def _install_fake_vjepa(monkeypatch: pytest.MonkeyPatch) -> TinyBackbone:
    src = types.ModuleType("src")
    src.__path__ = []
    models = types.ModuleType("src.models")
    models.__path__ = []
    vision_transformer = types.ModuleType("src.models.vision_transformer")

    def vit_large(**_kwargs):
        backbone = TinyBackbone(depth=24)
        backbone.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
        return backbone

    vision_transformer.vit_large = vit_large
    src.models = models
    models.vision_transformer = vision_transformer
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.models", models)
    monkeypatch.setitem(sys.modules, "src.models.vision_transformer", vision_transformer)
    return vit_large()


def test_vjepa_adapter_requires_and_reports_pretrained_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _install_fake_vjepa(monkeypatch)
    state = {f"module.encoder.{key}": value.detach().clone() for key, value in source.state_dict().items()}
    state["module.encoder.pos_embed"] = torch.zeros(1)
    checkpoint_path = tmp_path / "vjepa.pt"
    torch.save({"encoder": state}, checkpoint_path)

    encoder = VJEPA2Encoder(
        str(checkpoint_path),
        input_size=(480, 768),
        num_frames=16,
        output_layers=(8, 12, 16, 20, 24),
    )
    assert encoder.load_report.ready
    assert encoder.load_report.coverage == 1.0
    assert "pos_embed" in encoder.load_report.ignored_checkpoint_keys

    broken = dict(state)
    broken.pop("module.encoder.blocks.23.weight")
    torch.save({"encoder": broken}, tmp_path / "vjepa_broken.pt")
    with pytest.raises(RuntimeError, match="last_transformer_block|missing model keys"):
        VJEPA2Encoder(
            str(tmp_path / "vjepa_broken.pt"),
            input_size=(480, 768),
            num_frames=16,
            output_layers=(8, 12, 16, 20, 24),
        )


class FakeVideoMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.embed_dim = 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 8, 2))
        self.pos_drop = nn.Identity()


def test_videomaev2_adapter_validates_core_layers_and_allows_resolution_pos_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    models = types.ModuleType("models")
    models.__path__ = []
    modeling_finetune = types.ModuleType("models.modeling_finetune")
    modeling_finetune.vit_base_patch16_224 = lambda **_kwargs: FakeVideoMAE()
    models.modeling_finetune = modeling_finetune
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "models.modeling_finetune", modeling_finetune)

    source = FakeVideoMAE()
    state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    state["pos_embed"] = torch.zeros(1, 4, 2)
    checkpoint_path = tmp_path / "videomae.pt"
    torch.save({"model": state}, checkpoint_path)
    encoder = VideoMAEv2Encoder(
        str(checkpoint_path),
        source_root=str(tmp_path),
        input_size=(480, 768),
        num_frames=16,
    )
    assert encoder.load_report.ready
    assert encoder.load_report.allowed_shape_mismatches[0].key == "pos_embed"


@pytest.mark.integration
def test_wan_decoder_requires_complete_pretrained_decoder(tmp_path: Path):
    source_root = Path("/share/project/lgy/Wan2.2")
    if not (source_root / "wan/modules/vae2_2.py").is_file():
        pytest.skip("Pinned local Wan2.2 source is unavailable")

    uninitialized = WanVideoDecoder(
        source_root=str(source_root), base_dim=8, output_size=(32, 48), load_pretrained=False
    )
    state = {
        **{f"decoder.{key}": value.detach().clone() for key, value in uninitialized.decoder.state_dict().items()},
        **{f"conv2.{key}": value.detach().clone() for key, value in uninitialized.pre_decoder.state_dict().items()},
        "decoder.upsamples.0.upsamples.3.time_conv.weight": torch.zeros(1),
    }
    checkpoint_path = tmp_path / "wan.pt"
    torch.save(state, checkpoint_path)
    loaded = WanVideoDecoder(
        str(checkpoint_path), source_root=str(source_root), base_dim=8, output_size=(32, 48)
    )
    assert loaded.load_report.ready
    assert loaded.load_report.decoder.coverage == 1.0
    assert "upsamples.0.upsamples.3.time_conv.weight" in (
        loaded.load_report.decoder.ignored_checkpoint_keys
    )

    broken = {key: value for key, value in state.items() if key.startswith("conv2.") or key == "decoder.conv1.weight"}
    torch.save(broken, tmp_path / "wan_broken.pt")
    with pytest.raises(RuntimeError, match="missing model keys|missing critical groups"):
        WanVideoDecoder(
            str(tmp_path / "wan_broken.pt"),
            source_root=str(source_root),
            base_dim=8,
            output_size=(32, 48),
        )
