"""Methods and classes to deal with streaming synthetic data."""

import random
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from one_to_many_gan import GeneratorHandler
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from siamese.datasets import find_all_images


class ShoemarkImpressionType(IntEnum):
    """Enum to describe what kind of impression a returned tensor is."""

    SHOEMARK_NO_BACK = auto()  # Real shoemark with no background
    SHOEMARK_BACK = auto()  # Real shoemark with a background
    NO_SHOEMARK = auto()


@dataclass
class ShoemarkImpression:
    """Container to hold a shoemark and its type."""

    data: torch.Tensor
    impression_type: ShoemarkImpressionType


class StreamingDataset(Dataset):
    """Load random floor images and shoeprints."""

    def __init__(
        self,
        shoeprint_path: Path,
        real_shoemark_path: Path,
        floor_path: Path,
        image_size: tuple[int, int],
        min_floor_roi_height: int,
        synthetic_ratio: float,
    ):
        #  / "train" # Streaming is only used for training
        shoeprint_path = Path(shoeprint_path).expanduser()
        real_shoemark_path = Path(real_shoemark_path).expanduser()
        floor_path = Path(floor_path).expanduser()

        self.shoeprint_files = find_all_images(shoeprint_path)
        self.shoeprint_tensors = [
            F.to_tensor(Image.open(shoeprint_path).convert("RGB"))
            for shoeprint_path in self.shoeprint_files
        ]

        floor_files = find_all_images(floor_path)
        self.floor_tensors = [
            F.to_tensor(Image.open(floor_path).convert("RGB")) for floor_path in floor_files
        ]

        self.shoemark_tensors = defaultdict(list)

        def allocate_shoemarks(files: list[Path], impression_type: ShoemarkImpressionType):
            for f in files:
                class_id = int(f.stem.split("_")[0])
                self.shoemark_tensors[class_id].append(
                    ShoemarkImpression(F.to_tensor(Image.open(f).convert("RGB")), impression_type)
                )

        shoemark_no_back_files = find_all_images(real_shoemark_path / "no_back")
        allocate_shoemarks(shoemark_no_back_files, ShoemarkImpressionType.SHOEMARK_NO_BACK)

        shoemark_back_files = find_all_images(real_shoemark_path / "back")
        allocate_shoemarks(shoemark_back_files, ShoemarkImpressionType.SHOEMARK_BACK)

        self.image_size = image_size
        self.min_floor_roi_height = min_floor_roi_height
        self.synthetic_ratio = synthetic_ratio

    def __len__(self):
        return len(self.shoeprint_tensors)

    def __getitem__(self, idx):
        # TODO use defaultdict to store shoeprint tensors with IDs
        shoeprint_file = self.shoeprint_files[idx]
        shoeprint_id = int(shoeprint_file.stem.split("_")[0])
        shoeprint_image = self.shoeprint_tensors[idx]

        # For shoeprints that have a shoemark, we don't always want to use the real shoemark
        use_synthetic = random.random() > self.synthetic_ratio

        if not use_synthetic and shoeprint_id in self.shoemark_tensors:
            shoemark = random.choice(self.shoemark_tensors[shoeprint_id])
            shoemark_image = shoemark.data
            shoemark_image_type = shoemark.impression_type
        else:
            shoemark_image = torch.zeros_like(shoeprint_image)
            shoemark_image_type = ShoemarkImpressionType.NO_SHOEMARK

        floor_image = self._get_floor_image()

        # The shoemark_image_type will be used for masking shoemark_image and floor_image, as
        # these are not required for all shoeprints
        if (
            shoeprint_image.shape[1] != 512
            or floor_image.shape[1] != 512
            or shoemark_image.shape[1] != 512
        ):
            raise ValueError

        return shoeprint_image, floor_image, shoemark_image, torch.tensor(shoemark_image_type)

    def _get_floor_image(self):
        floor_image = random.choice(self.floor_tensors)

        # Randomly rotate by 0, 90, 180 or 270 degrees
        # TODO store rotated images in RAM
        # k = random.randint(0, 3)
        # floor_image = torch.rot90(floor_image, k=k, dims=[-2, -1])

        # Get tensor dimensions
        _, h, w = floor_image.shape

        # Width must be at least 2*height
        max_roi_height = h if w > (h // 2) else w * 2

        if max_roi_height < self.min_floor_roi_height:
            raise ValueError

        # Randomly select height
        roi_height = random.randint(self.min_floor_roi_height, max_roi_height)
        roi_width = roi_height // 2  # 2:1 aspect ratio

        # Randomly select top-left corner
        top = random.randint(0, h - roi_height)
        left = random.randint(0, w - roi_width)

        # Extract ROI
        roi = floor_image[:, top : top + roi_height, left : left + roi_width]

        # Scale to 512x256
        # Use cheap interpolation and no antialias as we are scaling down
        return F.resize(
            roi,
            size=self.image_size,  # pyright: ignore [reportArgumentType]
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )


class RandomCropAndPad:
    """Randomly crop a batch of tensors and then scale and pad back to original shape."""

    def __init__(self, fill: float = 0.0, min_edge: int = 64, size: tuple[int, int] = (512, 256)):
        self.random_crop = transforms.RandomCrop(size)
        self.fill = fill
        self.min_edge = min_edge
        self.height, self.width = size

    def __call__(self, shoemarks: torch.Tensor):
        new_height = int(torch.randint(self.min_edge, self.height, (1,)))
        new_width = int(torch.randint(self.min_edge, self.width, (1,)))

        self.random_crop.size = (new_height, new_width)
        cropped = self.random_crop(shoemarks)

        aspect_ratio = new_height / new_width
        if new_height > new_width:
            scaled_height = self.height
            scaled_width = int(self.height / aspect_ratio)

            pad_h = 0
            pad_w = self.width - scaled_width
        else:
            scaled_height = int(self.width * aspect_ratio)
            scaled_width = self.width

            pad_h = self.height - scaled_height
            pad_w = 0

        scaled = F.resize(cropped, (scaled_height, scaled_width))  # pyright: ignore [reportArgumentType]

        pad_top = (pad_h // 2) + (pad_h % 2)
        pad_left = (pad_w // 2) + (pad_w % 2)
        pad_bottom = pad_h // 2
        pad_right = pad_w // 2

        return F.pad(scaled, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)  # pyright: ignore [reportArgumentType]


def create_shoemarks(
    shoeprints: torch.Tensor,
    floor_images: torch.Tensor,
    shoemarks: torch.Tensor,
    shoemark_type_mask: torch.Tensor,
    generator_handler: GeneratorHandler,
    difficulty: float,
    streaming_transform: "StreamingTransforms",
    device,
):
    """Handle the creation of shoemarks for a batch of shoeprints."""
    indices = torch.arange(shoemarks.size(0), device=device)

    # Real shoemarks with a background that require no synthetic augmentation
    background_mask = shoemark_type_mask == ShoemarkImpressionType.SHOEMARK_BACK
    # Real shoemarks with no background that will require background substitution
    no_background_mask = shoemark_type_mask == ShoemarkImpressionType.SHOEMARK_NO_BACK
    # Shoeprints that don't have a real shoemark and so a synthetic one will be chosen
    no_shoemark_mask = shoemark_type_mask == ShoemarkImpressionType.NO_SHOEMARK

    # Used to ensure shoemark order
    background_indices = indices[background_mask]
    no_background_indices = indices[no_background_mask]
    no_shoemark_indices = indices[no_shoemark_mask]

    background_shoemarks = shoemarks[background_mask]
    no_background_shoemarks = shoemarks[no_background_mask]
    no_shoemark_shoeprints = shoeprints[no_shoemark_mask]

    # Generate shoemarks using shoeprints
    generated_shoemarks = generator_handler.generate(
        F.rgb_to_grayscale(no_shoemark_shoeprints), difficulty=difficulty, normalised=False
    )
    b, _, h, w = generated_shoemarks.shape
    generated_shoemarks = generated_shoemarks.expand(b, 3, h, w)

    # Floor images for shoemarks requiring them
    floor_images = floor_images[shoemark_type_mask != ShoemarkImpressionType.SHOEMARK_BACK]

    # Combine no_background and generated shoemarks
    no_background_shoemarks = torch.cat([no_background_shoemarks, generated_shoemarks])
    combined_indices = torch.cat([no_background_indices, no_shoemark_indices])

    # Determine which need background substitution
    include_background = no_background_shoemarks.std(dim=(1, 2, 3)) > 0.08

    # Randomly apply pre blend transforms
    pre_blend_transformed = torch.rand(no_background_shoemarks.shape[0]) > 0.5
    pre_blend_mask = torch.zeros(shoemarks.size(0), dtype=torch.bool, device=device)
    pre_blend_mask[combined_indices[pre_blend_transformed]] = True

    no_background_shoemarks[pre_blend_transformed] = streaming_transform.pre_blend_transforms(
        no_background_shoemarks[pre_blend_transformed]
    )

    # Convert some shoemarks to blue (enhanced blood)
    no_background_shoemarks[include_background] = streaming_transform.shoemark_blue_shift(
        no_background_shoemarks[include_background]
    )

    # Don't affine cropped shoemarks
    no_background_shoemarks[~pre_blend_transformed] = streaming_transform.shoemark_affine(
        no_background_shoemarks[~pre_blend_transformed]
    )

    synth_background_shoemarks = (
        no_background_shoemarks[include_background] * floor_images[: include_background.sum()]
    )

    # Build list of (index, shoemark) tuples
    result_pairs = []

    # Add background shoemarks
    for idx, shoemark in zip(background_indices.tolist(), background_shoemarks, strict=True):
        result_pairs.append((idx, shoemark))

    # Add synthetic background shoemarks
    synth_bg_iter = iter(synth_background_shoemarks)
    no_bg_iter = iter(no_background_shoemarks[~include_background])

    for idx, needs_bg in zip(combined_indices.tolist(), include_background.tolist(), strict=True):
        if needs_bg:
            result_pairs.append((idx, next(synth_bg_iter)))
        else:
            result_pairs.append((idx, next(no_bg_iter)))

    # Sort by original index and stack
    result_pairs.sort(key=lambda x: x[0])

    # We need to know which shoemarks have been pre-blend transformed as pre/post blending
    # transformations are mutually exclusive

    result_indices = torch.tensor([idx for idx, _ in result_pairs], device=device)
    sorted_pre_blended = pre_blend_mask[result_indices]

    return torch.stack([shoemark for _, shoemark in result_pairs]), sorted_pre_blended


class StreamingTransforms:
    """Transforms used in the streaming dataloader."""

    def __init__(
        self,
        fill: float = 0.0,
        min_edge: int = 64,
        size: tuple[int, int] = (512, 256),
    ):
        self.post_blend_transform = transforms.RandomApply(
            transforms=[RandomCropAndPad(fill=fill, min_edge=min_edge, size=size)], p=0.5
        )

        # TODO this may be too aggressive for shoeprints
        # TODO specify in config
        self.shoemark_affine = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=20,  # pyright: ignore [reportArgumentType]
                    translate=(0.1, 0.3),
                    scale=(0.75, 1.25),
                    fill=1.0,
                    shear=[0.1] * 4,
                ),
            ]
        )
        self.shoeprint_affine = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10,  # pyright: ignore [reportArgumentType]
                    translate=(0.05, 0.15),
                    scale=(0.9, 1.1),
                    fill=1.0,
                    shear=[0.05] * 4,
                ),
            ]
        )

        def shoemark_blue_shift(shoemark: torch.Tensor):
            shift_amount = torch.rand(shoemark.shape[0], device=shoemark.device) * 0.25
            shoemark[:, 2, :, :].add_(shift_amount[:, None, None])
            return shoemark

        self.shoemark_blue_shift = transforms.RandomApply(
            [shoemark_blue_shift, transforms.Lambda(lambda x: torch.clamp(x, 0, 1))], p=1 / 3
        )

        # TODO specify in config
        self.photometric_transforms = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                    ]
                ),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2)], p=0.5),
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )

        # TODO set in config
        max_erasing_ratio = 0.5
        self.pre_blend_transforms = transforms.RandomErasing(
            p=1, scale=(0.1, max_erasing_ratio), ratio=(0.3, 3.33), value=1.0
        )


class AdaptiveNormalisation:
    """Use Exponential Moving Average (EMA) for normalising a stream of data."""

    def __init__(
        self,
        initial_mean: torch.Tensor,
        initial_std: torch.Tensor,
        device,
        momentum: float = 0.99,
    ):
        self.mean = initial_mean.clone().to(device)
        self.std = initial_std.clone().to(device)
        self.momentum = momentum
        self.n_samples = 0

    def __call__(self, batch: torch.Tensor, *, update: bool = True):
        # Normalise with current statistics
        normalised = (batch - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        if update:
            batch_mean = batch.mean(dim=[0, 2, 3])
            batch_std = batch.std(dim=[0, 2, 3])

            # Start with a high momentum, decrease over time
            # Improves early training stability
            effective_momentum = min(self.momentum, self.n_samples / (self.n_samples + 1))

            self.mean = effective_momentum * self.mean + (1 - effective_momentum) * batch_mean
            self.std = effective_momentum * self.std + (1 - effective_momentum) * batch_std
            self.n_samples += batch.size(0)

        return normalised

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "momentum": self.momentum,
            "n_sampels": self.n_samples,
        }

    def load_state_dict(self, state_dict: dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.momentum = state_dict["momentum"]
        self.n_samples = state_dict["n_samples"]


class DifficultyScheduler:
    """EMA-based difficulty scheduler."""

    def __init__(
        self,
        margin: float,
        beta: float = 0.99,
        initial_difficulty: float = 0.1,
        min_difficulty=0.1,
        max_difficulty=1.0,
    ):
        self.margin = margin
        self.beta = beta
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

    def update(self, pos_dists: torch.Tensor, neg_dists: torch.Tensor):
        # Calculate how well sperated the pairs are
        # Higher violation_score means network is struggling
        violations = torch.clamp(pos_dists - neg_dists + self.margin, min=0)
        violation_score = violations.mean().item()

        # Normalise score by margin to get [0, 1] range
        normalised_score = min(violation_score / self.margin, 1.0)

        target_difficulty = 1.0 - normalised_score

        difficulty = self.beta * self.difficulty + (1 - self.beta) * target_difficulty
        # Clamp to valid range
        self.difficulty = max(self.min_difficulty, min(self.max_difficulty, difficulty))

    def get_difficulty(self):
        return self.difficulty
