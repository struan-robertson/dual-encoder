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
    gan_config: Path
    resume_checkpoint: Path | None


class _Streaming(TypedDict):
    floor_image_data_dir: Path
    shoeprint_data_dir: Path
    shoemark_data_dir: Path
    min_floor_roi_height: int
    synthetic_ratio: float


class _Data(TypedDict):
    shoeprint_val_dir: Path
    shoeprint_dataset_mean: tuple[float, float, float]
    shoeprint_dataset_std: tuple[float, float, float]
    shoemark_val_dir: Path
    shoemark_dataset_mean: tuple[float, float, float]
    shoemark_dataset_std: tuple[float, float, float]
    streaming: _Streaming
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

    config["data"]["streaming"]["shoeprint_data_dir"] = Path(
        config["data"]["streaming"]["shoeprint_data_dir"]
    )
    config["data"]["streaming"]["shoemark_data_dir"] = Path(
        config["data"]["streaming"]["shoemark_data_dir"]
    )
    config["data"]["shoeprint_val_dir"] = Path(config["data"]["shoeprint_val_dir"])
    config["data"]["shoemark_val_dir"] = Path(config["data"]["shoemark_val_dir"])
    config["data"]["wvu_data_dir"] = Path(config["data"]["wvu_data_dir"])
    config["data"]["fid_data_dir"] = Path(config["data"]["fid_data_dir"])
    config["training"]["gan_config"] = Path(config["training"]["gan_config"])
    config["data"]["streaming"]["floor_image_data_dir"] = Path(
        config["data"]["streaming"]["floor_image_data_dir"]
    )
    config["training"]["resume_checkpoint"] = (
        Path(config["training"]["resume_checkpoint"])  # pyright: ignore [reportArgumentType]
        if config["training"]["resume_checkpoint"] != ""
        else None
    )
    return config
