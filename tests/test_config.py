import pytest
from torch import nn

from progressive_videorae.model import factory
from progressive_videorae.config import load_training_bundle, load_yaml


def test_model_and_training_configs_resolve_from_project_root():
    model = load_yaml("configs/model/full_480p.yaml")
    assert model["encoder"]["input_height"] == 480
    assert model["encoder"]["input_width"] == 768
    assert model["encoder"]["variant"] == "vitl"
    assert model["decoder"]["base_dim"] == 256
    assert model["state"]["channels"] == 48
    bundle = load_training_bundle("configs/train/stage1b.yaml")
    assert bundle["training"]["max_steps"] == 90000
    assert bundle["model"]["state"]["num_sets"] == 64
    assert bundle["data"]["target_fps"] == 12.0
    assert bundle["training"]["prefix_schedule"] == "mixed_accumulation"
    assert bundle["training"]["full_microbatches_per_step"] == 4
    assert bundle["training"]["full_objective_weight"] == 1.0
    assert bundle["training"]["prefix_objective_weight"] == 1.0
    assert bundle["training"]["temporal_l1_weight"] == 0.1
    assert bundle["training"]["log_dir"] == (
        "/share/project/liujingyi/logs/waverae/progressive_video_rae/stage1"
    )

    evaluation = load_yaml("configs/eval/full_480p.yaml")
    assert evaluation["prefixes"] == [1, 4, 8, 16, 32, 48, 63, 64]

    vitg_bundle = load_training_bundle(
        "configs/train/stage1b.yaml",
        model_config_path="configs/model/full_480p_vitg.yaml",
    )
    assert vitg_bundle["model"]["encoder"]["variant"] == "vitg"
    assert vitg_bundle["model"]["encoder"]["embed_dim"] == 1408



class StubVJEPA2Encoder(nn.Module):
    def __init__(self, *, variant: str, **_kwargs):
        super().__init__()
        self.variant = variant
        self.embed_dim = 1024 if variant == "vitl" else 1408


class StubProjector(nn.Module):
    def __init__(self, *, input_dim: int, **_kwargs):
        super().__init__()
        self.input_dim = input_dim


class StubDecoder(nn.Module):
    def __init__(self, *, base_dim: int, **_kwargs):
        super().__init__()
        self.base_dim = base_dim


def test_factory_uses_backbone_embed_dim_and_supports_legacy_names(monkeypatch):
    monkeypatch.setattr(factory, "VJEPA2Encoder", StubVJEPA2Encoder)
    monkeypatch.setattr(factory, "CausalFrequencyProjector", StubProjector)
    monkeypatch.setattr(factory, "WanVideoDecoder", StubDecoder)
    config = load_yaml("configs/model/full_480p_vitg.yaml")

    model = factory.build_model(
        config, load_decoder_pretrained=False, validate_pretrained=False
    )
    assert model.encoder.variant == "vitg"
    assert model.projector.input_dim == 1408
    assert model.decoder.base_dim == 256
    assert model.repa_projection.out_channels == 1408

    config["encoder"]["name"] = "vjepa2_vitl16"
    config["encoder"].pop("variant")
    config["encoder"]["embed_dim"] = 1024
    config["encoder"]["output_layers"] = [8, 12, 16, 20, 24]
    model = factory.build_model(
        config, load_decoder_pretrained=False, validate_pretrained=False
    )
    assert model.encoder.variant == "vitl"


def test_factory_rejects_configured_embed_dim_mismatch(monkeypatch):
    monkeypatch.setattr(factory, "VJEPA2Encoder", StubVJEPA2Encoder)
    config = load_yaml("configs/model/full_480p_vitg.yaml")
    config["encoder"]["embed_dim"] = 1024
    with pytest.raises(ValueError, match="does not match"):
        factory.build_model(
            config, load_decoder_pretrained=False, validate_pretrained=False
        )
