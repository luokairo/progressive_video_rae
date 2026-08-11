from __future__ import annotations

from pathlib import Path

import pytest

from progressive_videorae.checksums import (
    sha256_file,
    sha256_sidecar_path,
    verify_checkpoint_sha256,
)


def test_missing_sidecar_is_created_atomically_and_second_check_is_read_only(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    digest = verify_checkpoint_sha256(checkpoint, create_missing_sidecar=True)
    sidecar = sha256_sidecar_path(checkpoint)

    assert digest == sha256_file(checkpoint)
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  weights.pt\n"
    assert verify_checkpoint_sha256(checkpoint, create_missing_sidecar=False) == digest


def test_changed_checkpoint_fails_without_overwriting_sidecar(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"original")
    digest = verify_checkpoint_sha256(checkpoint, create_missing_sidecar=True)
    sidecar = sha256_sidecar_path(checkpoint)
    original_sidecar = sidecar.read_text(encoding="utf-8")

    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_checkpoint_sha256(checkpoint, create_missing_sidecar=True)
    assert sidecar.read_text(encoding="utf-8") == original_sidecar
    assert sha256_file(checkpoint) != digest


def test_malformed_sidecar_fails_without_overwriting_it(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    sidecar = sha256_sidecar_path(checkpoint)
    sidecar.write_text("not-a-digest\n", encoding="utf-8")
    original_sidecar = sidecar.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid SHA256 sidecar"):
        verify_checkpoint_sha256(checkpoint, create_missing_sidecar=True)
    assert sidecar.read_text(encoding="utf-8") == original_sidecar


def test_sidecar_filename_mismatch_is_rejected(tmp_path: Path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    sidecar = sha256_sidecar_path(checkpoint)
    sidecar.write_text(f"{sha256_file(checkpoint)}  other.pt\n", encoding="utf-8")
    original_sidecar = sidecar.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="filename mismatch"):
        verify_checkpoint_sha256(checkpoint, create_missing_sidecar=True)
