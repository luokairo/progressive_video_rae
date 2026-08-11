from __future__ import annotations

import argparse
from pathlib import Path

from ..checksums import verify_checkpoint_sha256


EXPECTED = {
    "vjepa2": (("vjepa2/original/model.pth", "vjepa2/vitl/vitl.pt", "vjepa2/vitl/original/model.pth"), 1_000_000),
    "wan2.2": ("wan2.2/ti2v_5b/Wan2.2_VAE.pth", 1_000_000),
    "videomaev2": ("videomaev2/vitb/distill/vit_b_k710_dl_from_giant.pth", 1_000_000),
    "evaluation": ("evaluation/i3d_torchscript.pt", 100_000),
}



def validate_checkpoint_structure(model: str, path: Path) -> None:
    if model == "evaluation":
        try:
            import torch

            torch.jit.load(str(path), map_location="cpu")
        except Exception as exc:
            raise ValueError(f"Invalid TorchScript evaluation checkpoint: {exc}") from exc
        return
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for checkpoint key validation") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint root is not a state dictionary")
    if model == "vjepa2":
        nested = state.get("encoder", state.get("target_encoder", state))
        if not isinstance(nested, dict) or not any("patch_embed" in key for key in nested):
            raise ValueError("V-JEPA2 checkpoint has no patch embedding weights")
        if not any("blocks.0." in key for key in nested) or not any("blocks." in key for key in nested):
            raise ValueError("V-JEPA2 checkpoint has no transformer block weights")
    elif model == "wan2.2":
        nested = state.get("model", state) if isinstance(state, dict) else state
        if not isinstance(nested, dict) or not any(
            key.removeprefix("module.").removeprefix("model.").startswith("decoder.")
            for key in nested
        ):
            raise ValueError("Wan2.2 checkpoint has no decoder.* weights")
        if not any(
            key.removeprefix("module.").removeprefix("model.").startswith("conv2.")
            for key in nested
        ):
            raise ValueError("Wan2.2 checkpoint has no conv2.* input adapter weights")
    elif model == "videomaev2":
        nested = state.get("model", state.get("module", state))
        if not isinstance(nested, dict) or not any("patch_embed" in key for key in nested):
            raise ValueError("VideoMAEv2 checkpoint has no patch embedding weights")
        if not any("blocks.0." in key for key in nested) or not any("blocks." in key for key in nested):
            raise ValueError("VideoMAEv2 checkpoint has no transformer block weights")


def validate(root: Path, models: list[str], write_sha256: bool) -> None:
    failures = []
    for model in models:
        candidates, minimum_size = EXPECTED[model]
        if isinstance(candidates, str):
            candidates = (candidates,)
        path = next((root / candidate for candidate in candidates if (root / candidate).is_file()), None)
        if path is None:
            failures.append(f"missing: one of {', '.join(str(root / candidate) for candidate in candidates)}")
            continue
        size = path.stat().st_size
        if size < minimum_size:
            failures.append(f"too small ({size} bytes): {path}")
            continue
        try:
            digest = verify_checkpoint_sha256(
                path, create_missing_sidecar=write_sha256
            )
        except Exception as exc:
            failures.append(f"invalid SHA256 identity ({exc}): {path}")
        try:
            validate_checkpoint_structure(model, path)
        except Exception as exc:
            failures.append(f"invalid structure ({exc}): {path}")
            continue
        print(f"OK {model}: {path} ({size} bytes, sha256={digest})")
    if failures:
        raise SystemExit("Checkpoint validation failed:\n- " + "\n- ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/project/liujingyi/ckpts"))
    parser.add_argument("--model", choices=[*EXPECTED, "all"], default="all")
    parser.add_argument("--write-sha256", action="store_true")
    args = parser.parse_args()
    models = list(EXPECTED) if args.model == "all" else [args.model]
    validate(args.root, models, args.write_sha256)


if __name__ == "__main__":
    main()
