from __future__ import annotations

from pathlib import Path
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
    epoch: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "pretrained_load_report": (
                unwrap(model).pretrained_load_report()
                if hasattr(unwrap(model), "pretrained_load_report")
                else None
            ),
            "discriminator": unwrap(discriminator).state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "rng_state": rng_state(),
            "config": config,
            "upstream_commits": {
                "vjepa2": "204698b45b3712590f06245fbfba32d3be539812",
                "videomaev2": "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b",
                "nova": "63c5a724fc4e264e229a95c893184434f00c9413",
                "wan2.2": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            },
        },
        path,
    )


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
    saved_pretrained_report = checkpoint.get("pretrained_load_report")
    if saved_pretrained_report is not None and (
        not isinstance(saved_pretrained_report, dict)
        or saved_pretrained_report.get("ready") is not True
    ):
        raise RuntimeError(
            f"Training checkpoint {path} records an unsuccessful pretrained-weight preflight"
        )
    unwrap(model).load_state_dict(checkpoint["model"], strict=True)
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
