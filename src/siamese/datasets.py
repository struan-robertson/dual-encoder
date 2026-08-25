"""Load datasets using torch.utils.data.Dataset."""

from collections import defaultdict
from pathlib import Path
from typing import cast

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

_to_tensor = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)


def get_id(path: Path):
    """Get the class ID of a shoeprint or shoemark file."""
    split_str = path.stem.split("_")

    if len(split_str) == 3:
        split_str = split_str[:-1]

    return "_".join(split_str)


def find_all_images(path: Path):
    """Return a list of paths to images within a directory."""
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

    def __init__(self, path: Path | str):
        path = Path(path).expanduser()

        self.files = find_all_images(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file).convert("RGB")

        return _to_tensor(image), file


class LabeledCombinedDataset(Dataset):
    """Load shoeprint and shoemark images. Returns (shoeprint, (shoemarks)) tuples."""

    def __init__(
        self,
        shoeprint_path: Path | str,
        shoemark_path: Path | str,
    ):
        shoeprint_path = Path(shoeprint_path).expanduser()
        shoemark_path = Path(shoemark_path).expanduser()

        self.shoemark_path = shoemark_path
        self.shoeprint_files = find_all_images(shoeprint_path)

        shoemark_files = find_all_images(shoemark_path)
        shoemark_classes = defaultdict(list)

        for f in shoemark_files:
            class_id = get_id(f)
            shoemark_classes[class_id].append(f)

        self.shoemark_classes = shoemark_classes

    def __len__(self):
        return len(self.shoeprint_files)

    def __getitem__(self, idx: int):
        shoeprint = self.shoeprint_files[idx]
        shoeprint_class = get_id(shoeprint)
        shoeprint_image = Image.open(shoeprint).convert("RGB")

        shoeprint_image = _to_tensor(shoeprint_image)

        # For validation/testing we want to test all shoeprints for a shoemark
        shoemark_files = self.shoemark_classes[shoeprint_class]
        shoemark_images = tuple(
            _to_tensor(Image.open(f).convert("RGB")) for f in shoemark_files
        )

        return shoeprint_class, (shoeprint_image, shoemark_images)

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
