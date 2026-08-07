from pathlib import Path


def test_default_model_config_contains_no_low_resolution_branch():
    text = Path("configs/model/full_480p.yaml").read_text(encoding="utf-8")
    assert "input_height: 480" in text
    assert "input_width: 768" in text
    assert "240" not in text
    assert "384" not in text

