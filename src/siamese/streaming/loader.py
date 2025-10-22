"""CPU part of the streaming pipeline."""

import random
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset

from siamese.datasets import _to_tensor, find_all_images, get_id


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
        *,
        labelled: bool = False,
    ):
        #  / "train" # Streaming is only used for training
        shoeprint_path = Path(shoeprint_path).expanduser()
        real_shoemark_path = Path(real_shoemark_path).expanduser()
        floor_path = Path(floor_path).expanduser()

        def rotate_image(image: torch.Tensor):
            _0 = image
            _90 = torch.rot90(image, 1, dims=[-2, -1])
            _180 = torch.rot90(image, 2, dims=[-2, -1])
            _260 = torch.rot90(image, 3, dims=[-2, -1])
            return _0, _90, _180, _260

        floor_files = find_all_images(floor_path)
        floor_tensors = [
            _to_tensor(Image.open(floor_path).convert("RGB"))
            for floor_path in floor_files
        ]
        self.floor_tensors = [
            rotated for tensor in floor_tensors for rotated in rotate_image(tensor)
        ]

        self.shoeprint_tensors = defaultdict(list)
        shoeprint_files = find_all_images(shoeprint_path)
        for f in shoeprint_files:
            class_id = get_id(f)
            self.shoeprint_tensors[class_id].append(
                _to_tensor(Image.open(f).convert("RGB"))
            )
        self.shoeprint_classes = list(self.shoeprint_tensors.keys())

        self.shoemark_tensors = defaultdict(list)

        def allocate_shoemarks(
            files: list[Path], impression_type: ShoemarkImpressionType
        ):
            for f in files:
                class_id = get_id(f)
                self.shoemark_tensors[class_id].append(
                    ShoemarkImpression(
                        _to_tensor(Image.open(f).convert("RGB")), impression_type
                    )
                )

        shoemark_no_back_files = find_all_images(real_shoemark_path / "no_back")
        allocate_shoemarks(
            shoemark_no_back_files, ShoemarkImpressionType.SHOEMARK_NO_BACK
        )
        shoemark_back_files = find_all_images(real_shoemark_path / "back")
        allocate_shoemarks(shoemark_back_files, ShoemarkImpressionType.SHOEMARK_BACK)

        self.image_size = image_size
        self.min_floor_roi_height = min_floor_roi_height
        self.synthetic_ratio = synthetic_ratio
        self.labelled = labelled

    def __len__(self):
        return len(self.shoeprint_classes)

    def __getitem__(self, idx):
        shoeprint_class = self.shoeprint_classes[idx]

        shoeprint_image = random.choice(self.shoeprint_tensors[shoeprint_class])
        # Its not an issue if they might be the same, shoeprints may have only one image
        shoeprint_gen_image = random.choice(self.shoeprint_tensors[shoeprint_class])

        # For shoeprints that have a real shoemark, we don't always want to use it
        use_synthetic = random.random() < self.synthetic_ratio
        if not use_synthetic and shoeprint_class in self.shoemark_tensors:
            shoemark = random.choice(self.shoemark_tensors[shoeprint_class])
            shoemark_image = shoemark.data
            shoemark_image_type = shoemark.impression_type
        else:
            shoemark_image = torch.zeros_like(shoeprint_image)
            shoemark_image_type = ShoemarkImpressionType.NO_SHOEMARK

        floor_image = self._get_floor_image()

        # The shoemark_image_type will be used for masking shoemark_image and
        # floor_image, as these are not required for all shoeprints
        if (
            shoeprint_image.shape[1] != 512
            or floor_image.shape[1] != 512
            or shoemark_image.shape[1] != 512
        ):
            raise ValueError

        if self.labelled:
            return (
                shoeprint_image,
                shoeprint_gen_image,
                floor_image,
                shoemark_image,
                torch.tensor(shoemark_image_type),
                shoeprint_class,
            )

        return (
            shoeprint_image,
            shoeprint_gen_image,
            floor_image,
            shoemark_image,
            torch.tensor(shoemark_image_type),
        )

    def _get_floor_image(self):
        floor_image = random.choice(self.floor_tensors)

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
        return transforms.functional.resize(
            roi,
            size=self.image_size,  # pyright: ignore [reportArgumentType]
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=False,
        )
