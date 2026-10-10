"""Resolve configured checkpoints relative to the Fill2SR checkout."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def checkpoint_paths(fill_dir=None, redux_dir=None, lora_path=None):
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["paths"]
    values = (
        fill_dir or config["flux_fill_dir"],
        redux_dir or config["flux_redux_dir"],
        lora_path or config["model_sr"],
    )
    paths = tuple(
        (PROJECT_ROOT / Path(value).expanduser()).resolve()
        for value in values
    )
    for path in paths[:2]:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing checkpoint directory: {path}. See README.md for download instructions.")
    if not paths[2].is_file():
        raise FileNotFoundError(f"Missing Ours-Full LoRA: {paths[2]}. See README.md for download instructions.")
    return paths
