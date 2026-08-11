from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a SHA256 digest from file contents without loading the file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Checkpoint is unavailable: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sidecar_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    return file_path.with_suffix(file_path.suffix + ".sha256")


def read_sha256_sidecar(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    sidecar = sha256_sidecar_path(file_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) not in (1, 2) or not _SHA256_PATTERN.fullmatch(fields[0]):
        raise ValueError(f"Invalid SHA256 sidecar: {sidecar}")
    if len(fields) == 2 and fields[1].lstrip("*") != file_path.name:
        raise ValueError(
            f"SHA256 sidecar filename mismatch: {fields[1]} != {file_path.name}"
        )
    return fields[0].lower()


def _write_sha256_sidecar_atomic(file_path: Path, digest: str) -> Path:
    sidecar = sha256_sidecar_path(file_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(f"{digest}  {file_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def verify_checkpoint_sha256(
    path: str | Path,
    *,
    create_missing_sidecar: bool,
) -> str:
    """Return the content digest, creating only a missing sidecar when authorized."""

    file_path = Path(path).expanduser().resolve()
    actual = sha256_file(file_path)
    sidecar = sha256_sidecar_path(file_path)
    if not sidecar.exists():
        if not create_missing_sidecar:
            raise FileNotFoundError(f"Missing SHA256 sidecar: {sidecar}")
        try:
            _write_sha256_sidecar_atomic(file_path, actual)
        except FileExistsError:
            pass
        else:
            return actual
    expected = read_sha256_sidecar(file_path)
    if expected != actual:
        raise RuntimeError(
            f"SHA256 mismatch for {file_path}: sidecar={expected}, actual={actual}"
        )
    return actual
