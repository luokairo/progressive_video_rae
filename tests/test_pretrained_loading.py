from __future__ import annotations

from pathlib import Path
import os
import sys
import types

import pytest
import torch
from torch import nn

from progressive_videorae.model.pretrained import load_validated_pretrained
from progressive_videorae.model.encoders.vjepa2 import VJEPA2Encoder
from progressive_videorae.model.encoders.videomaev2 import VideoMAEv2Encoder
from progressive_videorae.model.wan_decoder import WanVideoDecoder


class TinyBackbone(nn.Module):
    def __init__(self, depth: int = 3, embed_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
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

    partial = {key: value for key, value in checkpoint.items() if key.startswith("patch_embed.")}
    with pytest.raises(RuntimeError, match="coverage"):
        load_validated_pretrained(
            TinyBackbone(),
            partial,
            component="toy",
            checkpoint_path=tmp_path / "partial.pt",
            required_groups={"last": ("blocks.2.*",)},
            minimum_coverage=0.95,
        )


def _install_fake_vjepa(monkeypatch: pytest.MonkeyPatch) -> dict[str, TinyBackbone]:
    src = types.ModuleType("src")
    src.__path__ = []
    models = types.ModuleType("src.models")
    models.__path__ = []
    vision_transformer = types.ModuleType("src.models.vision_transformer")

    def vit_large(**_kwargs):
        return TinyBackbone(depth=24, embed_dim=1024)

    def vit_giant_xformers(**_kwargs):
        return TinyBackbone(depth=40, embed_dim=1408)

    vision_transformer.vit_large = vit_large
    vision_transformer.vit_giant_xformers = vit_giant_xformers
    src.models = models
    models.vision_transformer = vision_transformer
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.models", models)
    monkeypatch.setitem(sys.modules, "src.models.vision_transformer", vision_transformer)
    return {"vitl": vit_large(), "vitg": vit_giant_xformers()}


def test_vjepa_adapter_requires_and_reports_pretrained_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sources = _install_fake_vjepa(monkeypatch)
    state = {
        f"module.encoder.{key}": value.detach().clone()
        for key, value in sources["vitl"].state_dict().items()
    }
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

    broken = {
        key: value for key, value in state.items() if key.startswith("module.encoder.patch_embed.")
    }
    broken["module.encoder.pos_embed"] = state["module.encoder.pos_embed"]
    torch.save({"encoder": broken}, tmp_path / "vjepa_broken.pt")
    with pytest.raises(RuntimeError, match="coverage"):
        VJEPA2Encoder(
            str(tmp_path / "vjepa_broken.pt"),
            input_size=(480, 768),
            num_frames=16,
            output_layers=(8, 12, 16, 20, 24),
        )


def test_vjepa_vitg_uses_giant_builder_and_target_encoder_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sources = _install_fake_vjepa(monkeypatch)
    state = {
        f"module.encoder.backbone.{key}": value.detach().clone()
        for key, value in sources["vitg"].state_dict().items()
    }
    checkpoint_path = tmp_path / "vjepa_vitg.pt"
    torch.save({"state_dict": {"target_encoder": state}}, checkpoint_path)

    encoder = VJEPA2Encoder(
        str(checkpoint_path),
        input_size=(480, 768),
        num_frames=16,
        variant="vitg",
        output_layers=(8, 16, 24, 32, 40),
    )
    assert encoder.variant == "vitg"
    assert encoder.depth == 40
    assert encoder.embed_dim == 1408
    assert encoder.layer_norms[0].normalized_shape == (1408,)
    assert encoder.load_report.ready


def test_vjepa_rejects_unknown_variant_and_out_of_range_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_vjepa(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported V-JEPA2 variant"):
        VJEPA2Encoder(str(tmp_path / "unused.pt"), variant="vitG")
    with pytest.raises(ValueError, match="1-based indices"):
        VJEPA2Encoder(
            str(tmp_path / "unused.pt"), variant="vitg", output_layers=(8, 41)
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("variant", "checkpoint_path", "output_layers"),
    [
        (
            "vitl",
            "/share/project/liujingyi/ckpts/vjepa2/vitl/original/model.pth",
            (8, 12, 16, 20, 24),
        ),
        (
            "vitg",
            "/share/project/liujingyi/ckpts/vjepa2/vitg/original/model.pth",
            (8, 16, 24, 32, 40),
        ),
    ],
)
def test_real_vjepa_checkpoint_load_smoke(variant, checkpoint_path, output_layers):
    if os.environ.get("PVR_RUN_LARGE_WEIGHT_LOAD_TESTS") != "1":
        pytest.skip("Set PVR_RUN_LARGE_WEIGHT_LOAD_TESTS=1 for multi-GB checkpoint loads")
    source_root = Path(
        "/share/project/liujingyi/progressive_video_rae/third_party/upstream/vjepa2"
    )
    if not source_root.is_dir() or not Path(checkpoint_path).is_file():
        pytest.skip("Pinned V-JEPA2 source or requested checkpoint is unavailable")
    encoder = VJEPA2Encoder(
        checkpoint_path,
        source_root=str(source_root),
        input_size=(480, 768),
        num_frames=16,
        variant=variant,
        output_layers=output_layers,
    )
    assert encoder.load_report.ready


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
    with pytest.raises(RuntimeError, match="coverage"):
        WanVideoDecoder(
            str(tmp_path / "wan_broken.pt"),
            source_root=str(source_root),
            base_dim=8,
            output_size=(32, 48),
        )
