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

from siamese.datasets import find_all_images


class ShoemarkImpressionType(IntEnum):
    """Enum to describe what kind of impression a returned tensor is."""

    SHOEMARK_NO_BACK = auto()  # Real shoemark with no background
    SHOEMARK_BACK = auto()  # Real shoemark with a background
    NO_SHOEMARK = auto()


@dataclass
class ShoemarkImpression:
    """Container to hold a shoemark and its type."""

    path: Path
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
        shoeprint_path = (
            Path(shoeprint_path).expanduser() / "train"
        )  # Streaming is only used for training
        real_shoemark_path = Path(real_shoemark_path).expanduser()
        floor_path = Path(floor_path).expanduser()

        self.shoeprint_files = find_all_images(shoeprint_path)
        self.floor_files = find_all_images(floor_path)

        self.shoemark_files = defaultdict(list)

        def allocate_shoemarks(files: list[Path], impression_type: ShoemarkImpressionType):
            for f in files:
                class_id = int(f.stem.split("_")[0])
                self.shoemark_files[class_id].append(ShoemarkImpression(f, impression_type))

        shoemark_no_back_files = find_all_images(real_shoemark_path / "no_back")
        allocate_shoemarks(shoemark_no_back_files, ShoemarkImpressionType.SHOEMARK_NO_BACK)

        shoemark_back_files = find_all_images(real_shoemark_path / "back")
        allocate_shoemarks(shoemark_back_files, ShoemarkImpressionType.SHOEMARK_BACK)

        self.image_size = image_size
        self.min_floor_roi_height = min_floor_roi_height
        self.synthetic_ratio = synthetic_ratio

    def __len__(self):
        return len(self.shoeprint_files)

    def __getitem__(self, idx):
        shoeprint_file = self.shoeprint_files[idx]
        shoeprint_id = int(shoeprint_file.stem.split("_")[0])

        shoeprint_image = Image.open(shoeprint_file).convert("RGB")
        shoeprint_image = F.to_tensor(shoeprint_image)

        # For shoeprints that have a shoemark, we don't always want to use the real shoemark
        use_synthetic = random.random() > self.synthetic_ratio

        if not use_synthetic and shoeprint_id in self.shoemark_files:
            shoemark = random.choice(self.shoemark_files[shoeprint_id])
            shoemark_image = Image.open(shoemark.path).convert("RGB")
            shoemark_image = F.to_tensor(shoemark_image)
            shoemark_image_type = shoemark.impression_type
        else:
            shoemark_image = torch.zeros_like(shoeprint_image)
            shoemark_image_type = ShoemarkImpressionType.NO_SHOEMARK

        if shoemark_image_type != ShoemarkImpressionType.SHOEMARK_BACK:
            floor_image = random.choice(self.floor_files)
            floor_image = self._transform_floor_image(floor_image)
        else:
            floor_image = torch.zeros_like(shoeprint_image)

        # The shoemark_image_type will be used for masking shoemark_image and floor_image, as
        # these are not required for all shoeprints
        return shoeprint_image, floor_image, shoemark_image, torch.tensor(shoemark_image_type)

    def _transform_floor_image(self, floor_image_path: Path):
        floor_image = Image.open(floor_image_path).convert("RGB")
        floor_image = F.to_tensor(floor_image)

        # Randomly rotate by 0, 90, 180 or 270 degrees
        k = random.randint(0, 3)
        floor_image = torch.rot90(floor_image, k=k, dims=[-2, -1])

        # Get tensor dimensions
        _, h, w = floor_image.shape

        # Width must be at least 2*height
        max_roi_height = min(h, w // 2)

        if max_roi_height < self.min_floor_roi_height:
            raise ValueError

        # Randomly select height
        roi_height = random.randint(self.min_floor_roi_height, max_roi_height)
        roi_width = roi_height * 2  # 2:1 aspect ratio

        # Randomly select top-left corner
        top = random.randint(0, h - roi_height)
        left = random.randint(0, w - roi_width)

        # Extract ROI
        roi = floor_image[:, top : top + roi_height, left : left + roi_width]

        # Scale to 512x256
        roi = roi.unsqueeze(0)
        roi = torch.nn.functional.interpolate(
            roi, size=self.image_size, mode="bilinear", align_corners=False
        )

        return roi.squeeze(0)


class RandomCropAndPad:
    """Randomly crop a batch of tensors and then scale and pad back to original shape."""

    def __init__(
        self, fill: float = 0.0, min_ratio: float = 0.25, size: tuple[int, int] = (512, 256)
    ):
        self.random_crop = transforms.RandomCrop(size)
        self.fill = fill
        self.min_ratio = min_ratio
        self.height, self.width = size

    def __call__(self, shoemarks: torch.Tensor):
        height_ratio = self.min_ratio + (1.0 - self.min_ratio) * torch.rand(1)
        width_ratio = self.min_ratio + (1.0 - self.min_ratio) * torch.rand(1)

        new_height = int(self.height * height_ratio)
        new_width = int(self.width * width_ratio)

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
):
    """Handle the creation of shoemarks for a batch of shoeprints."""
    indices = torch.arange(shoemarks.size(0))

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

    # Floor images for shoemarks requiring them
    floor_images = floor_images[shoemark_type_mask != ShoemarkImpressionType.SHOEMARK_BACK]

    # Generate shoemarks using shoeprints
    generated_shoemarks = generator_handler.generate(
        no_shoemark_shoeprints, difficulty=difficulty, normalised=False
    )

    # Combine no_background and generated shoemarks
    no_background_shoemarks = torch.cat([no_background_shoemarks, generated_shoemarks])
    combined_indices = torch.cat([no_background_indices, no_shoemark_indices])

    # Determine which need background substitution
    include_background = no_background_shoemarks.std(dim=1) > 0.08

    synth_background_shoemarks = no_background_shoemarks[include_background] * floor_images

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
    return torch.stack([shoemark for _, shoemark in result_pairs])


class StreamingTransforms:
    """Transforms used in the streaming dataloader."""

    def __init__(
        self,
        fill: float = 0.0,
        min_ratio: float = 0.25,
        size: tuple[int, int] = (512, 256),
    ):
        self.post_blend_transform = transforms.RandomApply(
            transforms=[RandomCropAndPad(fill=fill, min_ratio=min_ratio, size=size)], p=0.5
        )

        # TODO this may be too aggressive for shoeprints
        self.universal_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=70,  # pyright: ignore [reportArgumentType]
                    translate=(0.1, 0.3),
                    scale=(1.0, 0.5),
                    fill=1.0,
                    shear=[0.1] * 4,
                ),
            ]
        )

        self.photometric_transforms = transforms.Compose(
            [
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, hue=0.3)], p=0.5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5
                ),
            ]
        )

        max_erasing_ratio = 1.0 - min_ratio
        single_erase = transforms.RandomErasing(
            p=1, scale=(0.1, max_erasing_ratio), ratio=(0.3, 3.33), value=1.0
        )
        multi_erase = transforms.Compose(
            [
                transforms.RandomErasing(
                    p=1, scale=(0.1, max_erasing_ratio), ratio=(1.5, 2.5), value=1.0
                ),
                transforms.RandomErasing(
                    p=1, scale=(0.1, max_erasing_ratio), ratio=(0.5, 1.5), value=1.0
                ),
            ]
        )

        # Maybe useful for training resilience against aspect ratio changes
        random_resize_crop = transforms.RandomResizedCrop(
            (512, 256), scale=(min_ratio * 2, 1.0), ratio=(0.45, 0.55)
        )

        self.pre_blend_transforms = transforms.Compose(
            [
                transforms.RandomApply([single_erase], p=0.5),
                transforms.RandomApply([multi_erase], p=0.5),
                transforms.RandomApply([random_resize_crop], p=0.5),
            ]
        )
