from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    options = []
    if relative_to is not None:
        options.append(relative_to / candidate)
    options.extend((Path.cwd() / candidate, project_root() / candidate))
    for option in options:
        if option.exists():
            return option.resolve()
    return (project_root() / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install omegaconf or pyyaml to load project configurations") from exc
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected mapping in YAML config: {resolved}")
        return loaded
    config = OmegaConf.load(resolved)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def load_training_bundle(path: str | Path) -> dict[str, Any]:
    training_path = resolve_config_path(path)
    training = load_yaml(training_path)
    model_path = resolve_config_path(training.pop("model_config"), relative_to=training_path.parent)
    data_path = resolve_config_path(training.pop("data_config"), relative_to=training_path.parent)
    return {"training": training, "model": load_yaml(model_path), "data": load_yaml(data_path)}
