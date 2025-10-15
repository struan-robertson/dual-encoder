"""GPU transformations for streaming data."""

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F


class RandomCropAndPad:
    """Randomly crop a batch of tensors and then scale and pad back to original shape."""

    def __init__(
        self, fill: float = 0.0, min_edge: int = 64, size: tuple[int, int] = (512, 256)
    ):
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

        return F.pad(
            scaled,
            padding=(pad_left, pad_top, pad_right, pad_bottom),  # pyright: ignore [reportArgumentType]
            fill=self.fill,
        )


class StreamingTransforms:
    """Transforms used in the streaming dataloader."""

    def __init__(
        self,
        fill: float = 0.0,
        min_edge: int = 64,
        size: tuple[int, int] = (512, 256),
    ):
        self.post_blend_transform = transforms.RandomApply(
            transforms=[RandomCropAndPad(fill=fill, min_edge=min_edge, size=size)],
            p=0.5,
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
            [shoemark_blue_shift, transforms.Lambda(lambda x: torch.clamp(x, 0, 1))],
            p=1 / 3,
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
                # transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.4, hue=0.2)], p=0.5
                # ),
                # transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )

        # TODO set in config
        max_erasing_ratio = 0.5
        self.pre_blend_transforms = transforms.RandomErasing(
            p=1, scale=(0.1, max_erasing_ratio), ratio=(0.3, 3.33), value=1.0
        )
