"""Load datasets using torch.utils.data.Dataset."""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

_dataset_mode = Literal["train", "test", "val", "aug_val"]


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

        jpg_files = list(path.rglob("*.jpg"))
        png_files = list(path.rglob("*.png"))

        self.files = jpg_files + png_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file).convert("RGB")

        return F.to_tensor(image), file


def export_augmented(dataset: IndividualDataset, gpu_transform, save_dir: Path):
    """Export augmented images to a directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    for image, file in tqdm(dataset):
        transformed = gpu_transform(image)
        torchvision.utils.save_image(transformed, save_dir / file.name)


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


class RandomOffsetTransormation:
    """Randomly shift the image in any direction and rotate."""

    def __init__(
        self,
        offset: tuple[int, int] = (64, 32),
        max_rotation: int = 10,
        scale_diff: float = 0.25,
    ):
        self.offset = offset
        self.max_rotation = max_rotation
        self.scale_diff = scale_diff

    # TODO check this actually works
    def __call__(self, img):
        angle_rad = torch.rand(1).item() * 2 * math.pi
        dy = int(self.offset[0] * math.sin(angle_rad))
        dx = int(self.offset[1] * math.cos(angle_rad))
        rotation = torch.empty(1).uniform_(-self.max_rotation, self.max_rotation).item()
        scale = torch.empty(1).uniform_(1 - self.scale_diff, 1 + self.scale_diff).item()

        return F.affine(
            img,
            angle=rotation,
            translate=[dx, dy],
            scale=scale,
            shear=0.0,  # pyright: ignore [reportArgumentType]
            fill=1.0,  # pyright: ignore [reportArgumentType]
        )


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

        self.shoeprint_files = list(shoeprint_path.rglob("*.jpg")) + list(
            shoeprint_path.rglob("*.png")
        )

        shoemark_files = list(shoemark_path.rglob("*.jpg")) + list(shoemark_path.rglob("*.png"))

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
