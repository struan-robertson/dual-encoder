"""Define typed config options."""

import tomllib
from pathlib import Path
from typing import TypedDict


class _Hyperparameters(TypedDict):
    p_val: int
    margin: float
    batch_size: int
    embedding_size: int
    triplet_swapping: bool


class _PreTraining(TypedDict):
    pre_trained: bool
    frozen: bool
    epoch_unfreeze: int
    defrost: int
    permafrost: int
    refreeze: bool
    gradient_checkpointing: bool


class _Training(TypedDict):
    seed: int
    epochs: int
    print_iter: int
    val_iter: int
    name: str
    gpu_number: int
    pre_training: _PreTraining
    resume_checkpoint: Path | None


class _Gan(TypedDict):
    enabled: bool
    backend: str
    compile: bool
    pool_curriculum: bool
    config: Path
    initial_difficulty: float
    max_difficulty: float
    peak_steps: int


class _AffineAugmentation(TypedDict):
    enabled: bool
    flip: bool
    degrees: int
    translate: tuple[float, float]
    scale: tuple[float, float]
    shear: float
    fill: float


class _BackgroundSubstitution(TypedDict):
    enabled: bool


class _DustInvert(TypedDict):
    enabled: bool


class _BlueShift(TypedDict):
    enabled: bool
    p: float
    max_shift: float


class _PreBlendErasing(TypedDict):
    enabled: bool
    p: float
    min_scale: float
    max_scale: float
    fill: float


class _PostBlendCrop(TypedDict):
    enabled: bool
    p: float
    min_edge: int
    fill: float


class _Photometric(TypedDict):
    enabled: bool
    blur_kernel: tuple[int, int]
    blur_sigma: tuple[float, float]
    sharpness_factor: float
    jitter_p: float
    brightness: float
    hue: float


class Augmentations(TypedDict):
    """Per-augmentation toggles and strengths."""

    shoeprint_affine: _AffineAugmentation
    shoemark_affine: _AffineAugmentation
    shoemark_back_affine: _AffineAugmentation
    background_substitution: _BackgroundSubstitution
    dust_invert: _DustInvert
    blue_shift: _BlueShift
    pre_blend_erasing: _PreBlendErasing
    post_blend_crop: _PostBlendCrop
    photometric: _Photometric


class _Streaming(TypedDict):
    floor_image_data_dir: Path
    shoeprint_data_dir: Path
    shoemark_data_dir: Path
    synthetic_shoemark_data_dir: Path | None
    min_floor_roi_height: int
    synthetic_ratio: float
    real_pairs_only: bool


class _Data(TypedDict):
    streaming: _Streaming
    val_dir: Path
    test_dir: Path
    wvu_data_dir: Path
    fid_data_dir: Path
    image_size: tuple[int, int]


class Config(TypedDict):
    """Config options used for training and running the model."""

    hyperparameters: _Hyperparameters
    training: _Training
    gan: _Gan
    augmentations: Augmentations
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
    config["data"]["val_dir"] = Path(config["data"]["val_dir"])
    config["data"]["test_dir"] = Path(config["data"]["test_dir"])
    config["data"]["wvu_data_dir"] = Path(config["data"]["wvu_data_dir"])
    config["data"]["fid_data_dir"] = Path(config["data"]["fid_data_dir"])
    config["gan"]["config"] = Path(config["gan"]["config"])
    config["gan"].setdefault("backend", "one_to_many_gan")
    config["gan"].setdefault("pool_curriculum", False)
    config["data"]["streaming"].setdefault("real_pairs_only", False)
    config["training"]["pre_training"].setdefault("gradient_checkpointing", False)
    config["data"]["streaming"]["synthetic_shoemark_data_dir"] = (
        Path(config["data"]["streaming"]["synthetic_shoemark_data_dir"])
        if config["data"]["streaming"].get("synthetic_shoemark_data_dir")
        else None
    )
    config["data"]["streaming"]["floor_image_data_dir"] = Path(
        config["data"]["streaming"]["floor_image_data_dir"]
    )
    config["training"]["resume_checkpoint"] = (
        Path(config["training"]["resume_checkpoint"])  # pyright: ignore [reportArgumentType]
        if config["training"]["resume_checkpoint"] != ""
        else None
    )
    return config
