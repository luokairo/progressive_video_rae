from __future__ import annotations

import hashlib
from pathlib import Path
import os
import random
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn

from ..checksums import verify_checkpoint_sha256

CHECKPOINT_SCHEMA_VERSION = 4
UPSTREAM_COMMITS = {
    "vjepa2": "204698b45b3712590f06245fbfba32d3be539812",
    "wan2.2": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
}
PREVIOUS_STAGE = {"stage1b": "stage1a", "stage2a": "stage1b", "stage2b": "stage2a"}
OBJECTIVE_BY_STAGE = {
    "stage1a": "full_repa",
    "stage1b": "nested_spectral_hrepa_full_anchor",
    "stage2a": "full_repa",
    "stage2b": "full_repa_stateful",
}

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


def _checkpoint_path_from_model(model: nn.Module, component: str) -> str | None:
    module = unwrap(model)
    if component == "encoder":
        report = getattr(getattr(module, "encoder", None), "load_report", None)
        return getattr(report, "checkpoint_path", None)
    report = getattr(getattr(module, "decoder", None), "load_report", None)
    return getattr(getattr(report, "decoder", None), "checkpoint_path", None)


def pretrained_checkpoint_paths(model: nn.Module) -> dict[str, Path]:
    module = unwrap(model)
    encoder_path = _checkpoint_path_from_model(module, "encoder")
    wan_path = _checkpoint_path_from_model(module, "wan")
    if encoder_path is None or wan_path is None:
        raise RuntimeError(
            "Pretrained load reports must expose encoder and Wan checkpoint paths"
        )
    return {
        "vjepa": Path(encoder_path).expanduser().resolve(),
        "wan": Path(wan_path).expanduser().resolve(),
    }


def verify_pretrained_checkpoint_hashes(
    model: nn.Module, *, create_missing_sidecars: bool
) -> dict[str, str]:
    return {
        f"{component}_checkpoint_sha256": verify_checkpoint_sha256(
            path, create_missing_sidecar=create_missing_sidecars
        )
        for component, path in pretrained_checkpoint_paths(model).items()
    }


def deterministic_state_sha256(
    state: dict[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    selected = [(key, state[key]) for key in sorted(state) if key.startswith(prefixes)]
    if not selected:
        raise RuntimeError(f"No checkpoint tensors found for prefixes {prefixes}")
    for key, value in selected:
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def representation_identity(
    model: nn.Module,
    *,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    module = unwrap(model)
    if not hasattr(module, "projector") or not hasattr(module, "decoder"):
        return {}
    encoder = getattr(module, "encoder", None)
    encoder_report = getattr(encoder, "load_report", None)
    identity = {
        "encoder_checkpoint": getattr(encoder_report, "checkpoint_path", None),
        "encoder_variant": getattr(encoder, "variant", None),
        "selected_vjepa_layers": list(getattr(encoder, "output_layers", ())),
        "layout_checksum": module.projector.layout_checksum,
        "layout_version": module.projector.layout_version,
        "codec_id": module.decoder.codec_id,
        "decoder_id": module.decoder.decoder_id,
    }
    if checkpoint_hashes is not None:
        required = {"vjepa_checkpoint_sha256", "wan_checkpoint_sha256"}
        if set(checkpoint_hashes) != required:
            raise ValueError(
                f"Verified checkpoint hashes must contain exactly {sorted(required)}"
            )
        identity.update(checkpoint_hashes)
    return identity


def learned_identity(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        "projector_state_sha256": deterministic_state_sha256(
            state, prefixes=("projector.",)
        ),
        "bridge_state_sha256": deterministic_state_sha256(
            state, prefixes=("decoder.temporal_adapter.",)
        ),
    }


def validate_checkpoint_transition(
    checkpoint: dict[str, Any],
    *,
    target_stage: str,
    target_objective_mode: str,
    mode: Literal["init", "resume"],
    allow_smoke_checkpoint: bool,
    selected_intermediate: bool = False,
) -> None:
    schema = int(checkpoint.get("checkpoint_schema_version", 0))
    if schema != CHECKPOINT_SCHEMA_VERSION:
        if not (allow_smoke_checkpoint and schema == 3):
            raise RuntimeError(
                f"Checkpoint schema v{schema} cannot be used for this run; expected v4"
            )
    if schema == CHECKPOINT_SCHEMA_VERSION:
        stage_max_steps = checkpoint.get("stage_max_steps")
        stage_complete = checkpoint.get("stage_complete")
        optimizer_step = checkpoint.get("optimizer_step")
        if not isinstance(stage_max_steps, int) or stage_max_steps <= 0:
            raise RuntimeError("Schema v4 checkpoint has invalid stage_max_steps")
        if not isinstance(optimizer_step, int) or optimizer_step < 0:
            raise RuntimeError("Schema v4 checkpoint has invalid optimizer_step")
        if not isinstance(stage_complete, bool):
            raise RuntimeError("Schema v4 checkpoint has invalid stage_complete")
        if stage_complete != (optimizer_step >= stage_max_steps):
            raise RuntimeError("Schema v4 checkpoint stage_complete is inconsistent")
    saved_stage = checkpoint.get("stage")
    saved_objective = checkpoint.get("objective_mode")
    if mode == "resume":
        if saved_stage != target_stage or saved_objective != target_objective_mode:
            raise RuntimeError("Resume checkpoint stage/objective mismatch")
        return
    expected_source = PREVIOUS_STAGE.get(target_stage)
    if expected_source is None:
        raise RuntimeError("Stage 1-A must start fresh from pretrained weights")
    if saved_stage != expected_source:
        raise RuntimeError(
            f"{target_stage} must initialize from {expected_source}, got {saved_stage}"
        )
    expected_objective = OBJECTIVE_BY_STAGE[expected_source]
    if saved_objective != expected_objective:
        raise RuntimeError(f"{expected_source} checkpoint objective mismatch")
    if schema == CHECKPOINT_SCHEMA_VERSION and not bool(checkpoint.get("stage_complete")):
        if not allow_smoke_checkpoint and not selected_intermediate:
            raise RuntimeError(
                f"{target_stage} requires a completed {expected_source} checkpoint"
            )


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
    static_identity: dict[str, Any] | None = None,
    run_id: str | None = None,
    checkpoint_dir: str | None = None,
    source_checkpoint: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state, excluded_model_prefixes = checkpoint_model_state(model)
    training_config = config.get("training") if isinstance(config, dict) else None
    stage_max_steps = (
        int(training_config["max_steps"])
        if isinstance(training_config, dict) and "max_steps" in training_config
        else None
    )
    identity = dict(static_identity or representation_identity(model))
    if identity:
        identity.update(learned_identity(model_state))
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
            "representation_identity": identity,
            "discriminator": unwrap(discriminator).state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage_max_steps": stage_max_steps,
            "stage_complete": (
                stage_max_steps is not None and optimizer_step >= stage_max_steps
            ),
            "stage": stage,
            "objective_mode": objective_mode,
            "discriminator_update_count": discriminator_update_count,
            "epoch": epoch,
            "log_file": log_file,
            "rng_state": rng_state(),
            "run_id": run_id,
            "checkpoint_dir": checkpoint_dir,
            "source_checkpoint": source_checkpoint,
            "config": config,
            "upstream_commits": UPSTREAM_COMMITS,
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
    static_identity: dict[str, Any] | None = None,
    target_stage: str | None = None,
    target_objective_mode: str | None = None,
    load_mode: Literal["init", "resume"] | None = None,
    allow_smoke_checkpoint: bool = False,
    selection_certificate: str | Path | None = None,
    mmap: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(
        str(Path(path).expanduser().resolve()),
        map_location="cpu",
        mmap=mmap,
        weights_only=False,
    )
    schema = int(checkpoint.get("checkpoint_schema_version", 0))
    if schema != CHECKPOINT_SCHEMA_VERSION and not (
        allow_smoke_checkpoint and schema == 3
    ):
        raise RuntimeError(
            f"Checkpoint schema v{schema} cannot be loaded; expected v4"
        )
    transition_values = (target_stage, target_objective_mode, load_mode)
    selected_intermediate = False
    if selection_certificate is not None:
        if load_mode != "init" or target_stage is None:
            raise ValueError("A selection certificate is only valid for --init-from")
        previous_stage = PREVIOUS_STAGE.get(str(target_stage))
        if previous_stage is None:
            raise ValueError("Stage 1-A cannot use a selection certificate")
        from .selection_certificate import validate_selection_certificate

        validate_selection_certificate(
            selection_certificate,
            source_checkpoint=path,
            target_stage=str(target_stage),
            previous_stage=previous_stage,
            previous_objective=OBJECTIVE_BY_STAGE[previous_stage],
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
        selected_intermediate = True
    if any(value is not None for value in transition_values):
        if not all(value is not None for value in transition_values):
            raise ValueError(
                "Checkpoint transition validation requires stage, objective and mode"
            )
        validate_checkpoint_transition(
            checkpoint,
            target_stage=str(target_stage),
            target_objective_mode=str(target_objective_mode),
            mode=load_mode,
            allow_smoke_checkpoint=allow_smoke_checkpoint,
            selected_intermediate=selected_intermediate,
        )

    expected_contract = unwrap(model).state_contract.to_dict()
    saved_contract = checkpoint.get("state_contract")
    if schema == 3:
        legacy_identity = checkpoint.get("representation_identity")
        if not isinstance(saved_contract, dict) or not isinstance(legacy_identity, dict):
            raise RuntimeError(
                f"Training checkpoint {path} cannot migrate an incomplete schema v3 contract"
            )
        checksum = legacy_identity.get("layout_checksum")
        layout_version = legacy_identity.get("layout_version")
        if (
            layout_version != expected_contract["layout_version"]
            or saved_contract.get("layout_version") != layout_version
            or checksum != expected_contract["layout_checksum"]
        ):
            raise RuntimeError(f"Training checkpoint {path} schema v3 layout identity mismatch")
        saved_contract = dict(saved_contract)
        saved_contract["prefix_indexing"] = expected_contract["prefix_indexing"]
        saved_contract["layout_checksum"] = checksum
    if saved_contract != expected_contract:
        raise RuntimeError(
            f"Training checkpoint {path} has a missing or incompatible StateContract"
        )
    if checkpoint.get("upstream_commits") != UPSTREAM_COMMITS:
        raise RuntimeError(f"Training checkpoint {path} upstream commit mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError(f"Training checkpoint {path} has no model state dictionary")
    saved_identity = checkpoint.get("representation_identity")
    if not isinstance(saved_identity, dict):
        raise RuntimeError(f"Training checkpoint {path} is missing representation identity")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        expected_identity = static_identity or representation_identity(model)
        mismatches = {
            key: (saved_identity.get(key), value)
            for key, value in expected_identity.items()
            if saved_identity.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Training checkpoint {path} static representation identity mismatch: {mismatches}"
            )
        checkpoint_learned = learned_identity(state)
        for key, value in checkpoint_learned.items():
            if saved_identity.get(key) != value:
                raise RuntimeError(f"Training checkpoint {path} {key} mismatch")
    else:
        legacy_identity = representation_identity(model)
        if saved_identity != legacy_identity:
            raise RuntimeError(f"Training checkpoint {path} legacy identity mismatch")

    saved_pretrained_report = checkpoint.get("pretrained_load_report")
    if saved_pretrained_report is not None and (
        not isinstance(saved_pretrained_report, dict)
        or saved_pretrained_report.get("ready") is not True
    ):
        raise RuntimeError(
            f"Training checkpoint {path} records an unsuccessful pretrained-weight preflight"
        )

    load_model_state(model, state)
    if schema == CHECKPOINT_SCHEMA_VERSION and saved_identity:
        loaded_state = checkpoint_model_state(model)[0]
        loaded_learned = learned_identity(loaded_state)
        if any(
            saved_identity.get(key) != value
            for key, value in loaded_learned.items()
        ):
            raise RuntimeError(f"Training checkpoint {path} learned identity changed after load")
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
