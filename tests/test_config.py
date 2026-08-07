from progressive_video_rae.config import load_training_bundle, load_yaml


def test_model_and_training_configs_resolve_from_project_root():
    model = load_yaml("configs/model/full_480p.yaml")
    assert model["encoder"]["input_height"] == 480
    assert model["encoder"]["input_width"] == 768
    assert model["state"]["channels"] == 48
    bundle = load_training_bundle("configs/train/stage1b.yaml")
    assert bundle["training"]["max_steps"] == 90000
    assert bundle["model"]["state"]["num_sets"] == 64
    assert bundle["data"]["target_fps"] == 12.0

