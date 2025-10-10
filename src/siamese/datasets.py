"""Load datasets using torch.utils.data.Dataset."""

import random
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import Literal, cast

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

_dataset_mode = Literal["train", "test", "val", "aug_val"]


def _find_all_images(path: Path):
    files = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
    if len(files) == 0:
        raise FileNotFoundError

    return cast(list[Path], files)


def calculate_stats(loader: torch.utils.data.DataLoader):
    """Calculate per-channel mean and std using explicit sum of squares."""
    num_channels = loader.dataset[0].shape[0]
    sum_pixels = torch.zeros(num_channels)
    sum_squares = torch.zeros(num_channels)
    total_pixels = 0

    for image in tqdm(loader):  # [B, C, H, W]
        flattened = image.flatten(start_dim=2)  # [B, C, H*W]

        # Accumulate statistics
        sum_pixels += flattened.sum(dim=(0, 2))
        sum_squares += (flattened**2).sum(dim=(0, 2))
        total_pixels += flattened.shape[0] * flattened.shape[2]

    # Final calculations
    mean = sum_pixels / total_pixels
    std = torch.sqrt((sum_squares / total_pixels) - (mean**2))

    return mean, std


class IndividualDataset(Dataset):
    """Load either shoeprint or shoemark images. Used for statistic calculations."""

    def __init__(self, path: Path | str, *, mode: _dataset_mode = "train"):
        path = Path(path).expanduser() / mode

        self.files = _find_all_images(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file).convert("RGB")

        return F.to_tensor(image), file


def gpu_transform(
    image_size: tuple[int, int],
    *,
    mean: float | tuple[float, float, float],
    std: float | tuple[float, float, float],
    offset: bool = False,
    offset_translation: tuple[int, int] = (64, 32),
    offset_max_rotation: int = 10,
    offset_scale_diff: float = 0.25,
    flip: bool = True,
    normalise: bool = True,
):
    """Initialise transforms for a dataset."""
    transform_list = [transforms.Resize(image_size)]

    if normalise:
        transform_list.append(transforms.Normalize(mean, std))  # pyright: ignore [reportArgumentType]

    if offset:
        transform_list.append(
            RandomOffsetTransormation(offset_translation, offset_max_rotation, offset_scale_diff)  # pyright: ignore [reportArgumentType]
        )

    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())  # pyright: ignore [reportArgumentType]

    return transforms.Compose(transform_list)


class LabeledCombinedDataset(Dataset):
    """Load shoeprint and shoemark images. Returns (shoeprint, (shoemarks)) tuples."""

    def __init__(
        self,
        shoeprint_path: Path | str,
        shoemark_path: Path | str,
        *,
        mode: _dataset_mode,
        shoeprint_transform,
        shoemark_transform,
    ):
        shoeprint_path = Path(shoeprint_path).expanduser()
        shoemark_path = Path(shoemark_path).expanduser()

        if mode != "test":
            shoeprint_path = shoeprint_path / mode
            shoemark_path = shoemark_path / mode

        self.shoeprint_files = _find_all_images(shoeprint_path)

        shoemark_files = _find_all_images(shoemark_path)
        shoemark_classes = defaultdict(list)

        for f in shoemark_files:
            class_id = int(f.stem.split("_")[0])
            shoemark_classes[class_id].append(f)

        self.shoemark_classes = shoemark_classes

        self.shoeprint_transform = shoeprint_transform
        self.shoemark_transform = shoemark_transform
        self.mode = mode

    def __len__(self):
        return len(self.shoeprint_files)

    def __getitem__(self, idx: int):
        shoeprint = self.shoeprint_files[idx]
        shoeprint_class = int(shoeprint.stem.split("_")[0])
        shoeprint_image = Image.open(shoeprint).convert("RGB")

        shoeprint = self.shoeprint_transform(shoeprint_image)

        # For validation/testing we want to test all shoeprints for a shoemark
        if self.mode in {"val", "aug_val", "test"}:
            shoemark_files = self.shoemark_classes[shoeprint_class]
            shoemarks = tuple(
                self.shoemark_transform(Image.open(f).convert("RGB")) for f in shoemark_files
            )

            return shoeprint_class, (shoeprint, shoemarks)

        shoemark_file = random.choice(self.shoemark_classes[shoeprint_class])
        shoemark = self.shoemark_transform(Image.open(shoemark_file).convert("RGB"))

        return shoeprint, shoemark

    # Used for validation/test datasets where we don't work in batches
    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        sample = self[self.current_idx]
        self.current_idx += 1
        return sample


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

        self.shoeprint_files = _find_all_images(shoeprint_path)
        self.floor_files = _find_all_images(floor_path)

        self.shoemark_files = defaultdict(list)

        def allocate_shoemarks(files: list[Path], impression_type: ShoemarkImpressionType):
            for f in files:
                class_id = int(f.stem.split("_")[0])
                self.shoemark_files[class_id].append(ShoemarkImpression(f, impression_type))

        shoemark_no_back_files = _find_all_images(real_shoemark_path / "no_back")
        allocate_shoemarks(shoemark_no_back_files, ShoemarkImpressionType.SHOEMARK_NO_BACK)

        shoemark_back_files = _find_all_images(real_shoemark_path / "back")
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
