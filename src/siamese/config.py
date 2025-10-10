"""Define typed config options."""

from pathlib import Path
from typing import TypedDict

import tomllib


class _Hyperparameters(TypedDict):
    p_val: int
    margin: float
    batch_size: int
    embedding_size: int
    triplet_swapping: bool


class _Augmentation(TypedDict):
    max_translation: tuple[int, int]
    max_rotation: int
    max_scale: float
    flip: bool


class _PreTraining(TypedDict):
    pre_trained: bool
    frozen: bool
    epoch_unfreeze: int
    defrost: int
    permafrost: int
    refreeze: bool


class _Training(TypedDict):
    seed: int
    epochs: int
    print_iter: int
    val_iter: int
    name: str
    gpu_number: int
    pre_training: _PreTraining
    shoemark_augmentation: _Augmentation
    shoeprint_augmentation: _Augmentation


class _Data(TypedDict):
    shoeprint_data_dir: Path
    shoeprint_dataset_mean: tuple[float, float, float]
    shoeprint_dataset_std: tuple[float, float, float]
    shoemark_data_dir: Path
    shoemark_dataset_mean: tuple[float, float, float]
    shoemark_dataset_std: tuple[float, float, float]
    wvu_data_dir: Path
    fid_data_dir: Path
    image_size: tuple[int, int]


class Config(TypedDict):
    """Config options used for training and running the model."""

    hyperparameters: _Hyperparameters
    training: _Training
    data: _Data


def load_config(path: Path | str):
    """Load a TOML file of hyperparameters into a dictionary."""
    path = Path(path)

    with path.open("rb") as f:
        config: Config = tomllib.load(f)  # type: ignore[assignment]

    config["data"]["shoeprint_data_dir"] = Path(config["data"]["shoeprint_data_dir"])
    config["data"]["shoemark_data_dir"] = Path(config["data"]["shoemark_data_dir"])
    config["data"]["wvu_data_dir"] = Path(config["data"]["wvu_data_dir"])
    config["data"]["fid_data_dir"] = Path(config["data"]["fid_data_dir"])

    return config
