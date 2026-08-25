"""Typed configuration for the siamese project.

Contract: defaults < config.toml < CLI overrides. Every key is declared in the
dataclasses below; an unknown key in the TOML or on the command line is a hard
error. CLI overrides use the dotted path of the field, values parsed as TOML
literals so booleans and lists work:

    python src/training.py final_combined.toml --training.epochs 7500 \
        --curriculum.enabled true --data.image_size '[512, 256]'

Fields defaulting to None (training.name, the data directories) are required:
loading fails unless the TOML or the CLI provides them. Empty-string paths
mean "not set" (resume_checkpoint, synthetic_shoemark_data_dir).
"""

import argparse
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints


@dataclass
class Hyperparameters:
    """Loss, batch, and embedding settings."""

    distance_norm: int = 2  # the p of the Lp embedding distance
    margin: float = 2.0  # alpha of the triplet loss
    batch_size: int = 96  # also the size of the negative-mining pool
    embedding_size: int = 128
    triplet_swapping: bool = True


@dataclass
class PreTraining:
    """ImageNet initialisation and the progressive unfreezing ladder."""

    pre_trained: bool = True
    frozen: bool = True
    unfreeze_cadence: int = 500  # epochs between unfreezing one stage
    ladder_depth: int = 2  # residual stages the ladder opens; 0 unfreezes all
    gradient_checkpointing: bool = False


@dataclass
class Training:
    """Run identity, schedule length, and checkpoint resumption."""

    pre_training: PreTraining = field(default_factory=PreTraining)
    seed: int = 4242
    epochs: int = 5000
    print_iter: int = 100
    val_iter: int = 500
    name: str | None = None
    gpu_number: int = 0
    resume_checkpoint: Path | None = None


@dataclass
class Curriculum:
    """Difficulty curriculum over the pooled synthetic marks.

    The dataset draws each batch's synthetic marks from the pool tree nearest
    the scheduled difficulty (sibling trees named <base>_d<level>), ramped
    linearly from initial_difficulty to max_difficulty over peak_steps epochs.
    """

    enabled: bool = False
    initial_difficulty: float = 0.2
    max_difficulty: float = 1.0
    peak_steps: int = 3750


@dataclass
class AffineAugmentation:
    """Random affine transform applied to one image domain."""

    enabled: bool = True
    flip: bool = True
    degrees: float = 10.0
    translate: tuple[float, float] = (0.05, 0.15)
    scale: tuple[float, float] = (0.9, 1.1)
    shear: float = 0.05
    fill: float = 1.0


@dataclass
class BackgroundSubstitution:
    """Compositing of no-background marks onto floor images."""

    enabled: bool = True


@dataclass
class DustInvert:
    """Random polarity inversion of dust marks."""

    enabled: bool = True


@dataclass
class BlueShift:
    """Blue-channel shift simulating enhanced blood marks."""

    enabled: bool = True
    p: float = 0.333
    max_shift: float = 0.25


@dataclass
class PreBlendErasing:
    """Random erasing applied before background blending."""

    enabled: bool = True
    p: float = 0.5
    min_scale: float = 0.1
    max_scale: float = 0.5
    fill: float = 1.0


@dataclass
class PostBlendCrop:
    """Random crop, scale, and pad applied after background blending."""

    enabled: bool = True
    p: float = 0.5
    min_edge: int = 225
    fill: float = 0.0


@dataclass
class Photometric:
    """Blur, sharpening, and colour jitter of the capture stage."""

    enabled: bool = True
    blur_kernel: tuple[int, int] = (5, 9)
    blur_sigma: tuple[float, float] = (0.1, 5.0)
    sharpness_factor: float = 2.0
    jitter_p: float = 0.5
    brightness: float = 0.4
    hue: float = 0.2


@dataclass
class Augmentations:
    """Per-augmentation toggles and strengths."""

    shoeprint_affine: AffineAugmentation = field(default_factory=AffineAugmentation)
    shoemark_affine: AffineAugmentation = field(
        default_factory=lambda: AffineAugmentation(
            degrees=20.0, translate=(0.1, 0.3), scale=(0.75, 1.25), shear=0.1
        )
    )
    shoemark_back_affine: AffineAugmentation = field(
        default_factory=lambda: AffineAugmentation(fill=0.0)
    )
    background_substitution: BackgroundSubstitution = field(
        default_factory=BackgroundSubstitution
    )
    dust_invert: DustInvert = field(default_factory=DustInvert)
    blue_shift: BlueShift = field(default_factory=BlueShift)
    pre_blend_erasing: PreBlendErasing = field(default_factory=PreBlendErasing)
    post_blend_crop: PostBlendCrop = field(default_factory=PostBlendCrop)
    photometric: Photometric = field(default_factory=Photometric)


@dataclass
class Streaming:
    """Data directories and real/synthetic mixing for the training stream."""

    floor_image_data_dir: Path | None = None
    shoeprint_data_dir: Path | None = None
    shoemark_data_dir: Path | None = None
    synthetic_shoemark_data_dir: Path | None = None
    min_floor_roi_height: int = 512
    synthetic_ratio: float = 0.5
    real_pairs_only: bool = False


@dataclass
class Data:
    """Dataset locations and image geometry."""

    streaming: Streaming = field(default_factory=Streaming)
    val_dir: Path | None = None
    test_dir: Path | None = None
    wvu_data_dir: Path | None = None
    image_size: tuple[int, int] = (512, 256)


@dataclass
class Config:
    """Every option used for training and evaluation."""

    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    training: Training = field(default_factory=Training)
    curriculum: Curriculum = field(default_factory=Curriculum)
    augmentations: Augmentations = field(default_factory=Augmentations)
    data: Data = field(default_factory=Data)


_REQUIRED = (
    "training.name",
    "data.streaming.floor_image_data_dir",
    "data.streaming.shoeprint_data_dir",
    "data.streaming.shoemark_data_dir",
    "data.val_dir",
    "data.test_dir",
    "data.wvu_data_dir",
)


def _coerce(annotation, value, path):
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        if value == "" or value is None:
            return None
        annotation = next(a for a in get_args(annotation) if a is not type(None))
        origin = get_origin(annotation)
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            msg = f"{path}: expected a path string, got {value!r}"
            raise TypeError(msg)
        return Path(value).expanduser()
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            msg = f"{path}: expected a list, got {value!r}"
            raise TypeError(msg)
        return tuple(value)
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation is int and isinstance(value, bool):
        msg = f"{path}: expected int, got bool"
        raise TypeError(msg)
    if annotation in (bool, int, float, str) and not isinstance(value, annotation):
        msg = f"{path}: expected {annotation.__name__}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _apply(obj, mapping, path=""):
    hints = get_type_hints(type(obj))
    valid = {f.name for f in fields(obj)}
    for key, value in mapping.items():
        if key not in valid:
            msg = f"unknown config key: {path}{key}"
            raise KeyError(msg)
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                msg = f"{path}{key}: expected a table"
                raise TypeError(msg)
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, _coerce(hints[key], value, f"{path}{key}"))


def _get(config, dotted):
    obj = config
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _validate(config: Config):
    missing = [dotted for dotted in _REQUIRED if _get(config, dotted) is None]
    if missing:
        msg = "missing required config keys: " + ", ".join(missing)
        raise ValueError(msg)


def load_config(path: Path | str) -> Config:
    """Load a TOML file over the schema defaults; unknown keys are errors."""
    config = Config()
    with Path(path).open("rb") as f:
        _apply(config, tomllib.load(f))
    _validate(config)
    return config


def _leaf_paths(cls, prefix=""):
    hints = get_type_hints(cls)
    for f in fields(cls):
        if is_dataclass(hints[f.name]):
            yield from _leaf_paths(hints[f.name], f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}"


def parse_config(argv=None, default_config="config.toml") -> Config:
    """Config path plus dotted-path CLI overrides (--training.epochs 7500)."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("config", nargs="?", default=default_config,
                        help="config TOML (default: %(default)s)")
    for dotted in _leaf_paths(Config):
        parser.add_argument(f"--{dotted}", dest=dotted, metavar="VALUE")
    args = parser.parse_args(argv)

    config = Config()
    with Path(args.config).open("rb") as f:
        _apply(config, tomllib.load(f))
    for dotted, raw in vars(args).items():
        if dotted == "config" or raw is None:
            continue
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # bare strings need no quoting on the command line
        *parents, leaf = dotted.split(".")
        _apply(_get(config, ".".join(parents)) if parents else config, {leaf: value},
               f"{'.'.join(parents)}." if parents else "")
    _validate(config)
    return config
