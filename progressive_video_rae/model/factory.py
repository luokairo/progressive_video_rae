from __future__ import annotations

from typing import Any

from .encoders import VJEPA2Encoder, VideoMAEv2Encoder
from .model import ProgressiveVideoRAE
from .projector import CausalFrequencyProjector
from .wan_decoder import WanVideoDecoder


def build_model(
    config: dict[str, Any],
    *,
    load_decoder_pretrained: bool = True,
    validate_pretrained: bool = True,
) -> ProgressiveVideoRAE:
    video = config["video"]
    encoder_config = config["encoder"]
    state = config["state"]
    projector_config = config["projector"]
    decoder_config = config["decoder"]

    encoder_checkpoint = encoder_config.get("checkpoint_path")
    if not isinstance(encoder_checkpoint, str) or not encoder_checkpoint.strip():
        raise ValueError("encoder.checkpoint_path is required for pretrained encoder loading")
    decoder_checkpoint = decoder_config.get("checkpoint_path")
    if load_decoder_pretrained and (
        not isinstance(decoder_checkpoint, str) or not decoder_checkpoint.strip()
    ):
        raise ValueError("decoder.checkpoint_path is required for pretrained Wan2.2 loading")

    common_encoder = dict(
        checkpoint_path=encoder_checkpoint,
        source_root=encoder_config.get("source_root"),
        input_size=(encoder_config["input_height"], encoder_config["input_width"]),
        num_frames=video["num_frames"],
        patch_size=encoder_config["patch_size"],
        tubelet_size=encoder_config["tubelet_size"],
        output_layers=encoder_config["output_layers"],
        freeze=encoder_config.get("freeze", True),
    )
    if encoder_config["name"] == "vjepa2_vitl16":
        encoder = VJEPA2Encoder(
            **common_encoder,
            handle_nonsquare_inputs=encoder_config.get("handle_nonsquare_inputs", True),
        )
    elif encoder_config["name"] == "videomaev2_vitb":
        encoder = VideoMAEv2Encoder(
            **common_encoder,
            variant=encoder_config.get("variant", "vit_base_patch16_224"),
        )
    else:
        raise ValueError(f"Unsupported encoder: {encoder_config['name']}")

    projector = CausalFrequencyProjector(
        input_dim=encoder_config["embed_dim"],
        hidden_dim=projector_config["hidden_dim"],
        output_dim=state["channels"],
        num_frames=state["num_frames"],
        input_frames=video["num_frames"] // encoder_config["tubelet_size"],
        height=state["height"],
        width=state["width"],
        depth=projector_config["depth"],
        num_heads=projector_config["num_heads"],
        mlp_ratio=projector_config["mlp_ratio"],
        dropout=projector_config["dropout"],
        layout_version=state["layout_version"],
    )
    decoder = WanVideoDecoder(
        checkpoint_path=decoder_config.get("checkpoint_path"),
        source_root=decoder_config.get("source_root"),
        latent_channels=decoder_config["latent_channels"],
        base_dim=decoder_config["base_dim"],
        output_size=(decoder_config["output_height"], decoder_config["output_width"]),
        load_pretrained=load_decoder_pretrained,
    )
    model = ProgressiveVideoRAE(
        encoder,
        projector,
        decoder,
        encoder_dim=encoder_config["embed_dim"],
        decoder_feature_dim=decoder_config["base_dim"] * 4,
    )
    if validate_pretrained:
        model.assert_pretrained_ready()
    return model
