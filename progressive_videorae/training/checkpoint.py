from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Any

import numpy as np
import torch
from torch import nn


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_model_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[str]]:
    module = unwrap(model)
    state = module.state_dict()
    encoder = getattr(module, "encoder", None)
    if encoder is None or not getattr(encoder, "frozen", False):
        return state, []
    excluded_prefixes = ["encoder.backbone."]
    filtered = {
        key: value
        for key, value in state.items()
        if not any(key.startswith(prefix) for prefix in excluded_prefixes)
    }
    return filtered, excluded_prefixes


def representation_identity(model: nn.Module) -> dict[str, Any]:
    module = unwrap(model)
    if not hasattr(module, "projector") or not hasattr(module, "decoder"):
        return {}
    encoder = getattr(module, "encoder", None)
    encoder_report = getattr(encoder, "load_report", None)
    return {
        "encoder_checkpoint": getattr(encoder_report, "checkpoint_path", None),
        "encoder_variant": getattr(encoder, "variant", None),
        "selected_vjepa_layers": list(getattr(encoder, "output_layers", ())),
        "layout_checksum": module.projector.layout_checksum,
        "layout_version": module.projector.layout_version,
        "codec_id": module.decoder.codec_id,
        "decoder_id": module.decoder.decoder_id,
    }


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    module = unwrap(model)
    incompatible = module.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("encoder.backbone.",)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Training checkpoint model state is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )


def load_decoder_from_checkpoint(path: str | Path, *, model: nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_contract = unwrap(model).state_contract.to_dict()
    if checkpoint.get("state_contract") != expected_contract:
        raise RuntimeError("Decoder checkpoint StateContract is missing or incompatible")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError(f"Training checkpoint {path} has no model state dictionary")
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise RuntimeError(f"Training checkpoint {path} contains no decoder weights")
    unwrap(model).decoder.load_state_dict(decoder_state, strict=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: Any,
    discriminator_scheduler: Any,
    optimizer_step: int,
    discriminator_update_count: int,
    epoch: int,
    config: dict[str, Any],
    log_file: str | None = None,
    stage: str | None = None,
    objective_mode: str | None = None,
    update_latest: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state, excluded_model_prefixes = checkpoint_model_state(model)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "model": model_state,
            "excluded_model_prefixes": excluded_model_prefixes,
            "pretrained_load_report": (
                unwrap(model).pretrained_load_report()
                if hasattr(unwrap(model), "pretrained_load_report")
                else None
            ),
            "state_contract": unwrap(model).state_contract.to_dict(),
            "representation_identity": representation_identity(model),
            "discriminator": unwrap(discriminator).state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "checkpoint_schema_version": 3,
            "stage": stage,
            "objective_mode": objective_mode,
            "discriminator_update_count": discriminator_update_count,
            "epoch": epoch,
            "log_file": log_file,
            "rng_state": rng_state(),
            "config": config,
            "upstream_commits": {
                "vjepa2": "204698b45b3712590f06245fbfba32d3be539812",
                "wan2.2": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            },
        },
        temporary_path,
    )
    os.replace(temporary_path, path)
    if update_latest:
        latest = path.parent / "latest.pt"
        temporary_latest = path.parent / f".latest.{os.getpid()}.tmp"
        temporary_latest.unlink(missing_ok=True)
        temporary_latest.symlink_to(path.name)
        os.replace(temporary_latest, latest)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    discriminator: nn.Module | None = None,
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    generator_scheduler: Any | None = None,
    discriminator_scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_contract = unwrap(model).state_contract.to_dict()
    if checkpoint.get("state_contract") != expected_contract:
        raise RuntimeError(
            f"Training checkpoint {path} has a missing or incompatible StateContract"
        )
    saved_identity = checkpoint.get("representation_identity")
    if saved_identity is not None and saved_identity != representation_identity(model):
        raise RuntimeError(f"Training checkpoint {path} representation identity mismatch")
    saved_pretrained_report = checkpoint.get("pretrained_load_report")
    if saved_pretrained_report is not None and (
        not isinstance(saved_pretrained_report, dict)
        or saved_pretrained_report.get("ready") is not True
    ):
        raise RuntimeError(
            f"Training checkpoint {path} records an unsuccessful pretrained-weight preflight"
        )
    load_model_state(model, checkpoint["model"])
    if discriminator is not None and "discriminator" in checkpoint:
        unwrap(discriminator).load_state_dict(checkpoint["discriminator"], strict=True)
    if generator_optimizer is not None:
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    if discriminator_optimizer is not None:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    if generator_scheduler is not None:
        generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
    if discriminator_scheduler is not None:
        discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
