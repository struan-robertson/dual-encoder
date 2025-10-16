"""GPU transformations for streaming data."""

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from one_to_many_gan import GeneratorHandler

from siamese.streaming import ShoemarkImpressionType


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
        self.shoemark_back_affine = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10,  # pyright: ignore [reportArgumentType]
                    translate=(0.05, 0.15),
                    scale=(0.9, 1.1),
                    fill=0.0,
                    shear=[0.05] * 4,
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
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, hue=0.2)], p=0.5
                ),
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            ]
        )

        # TODO set in config
        max_erasing_ratio = 0.5
        self.pre_blend_transforms = transforms.RandomErasing(
            p=1, scale=(0.1, max_erasing_ratio), ratio=(0.3, 3.33), value=1.0
        )


def shoemark_pipeline(
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

    if len(no_shoemark_shoeprints) > 0:
        # Generate shoemarks using shoeprints
        generated_shoemarks = generator_handler.generate(
            F.rgb_to_grayscale(no_shoemark_shoeprints),
            difficulty=difficulty,
            normalised=False,
        )
        b, _, h, w = generated_shoemarks.shape
        generated_shoemarks = generated_shoemarks.expand(b, 3, h, w)

        # Combine no_background and generated shoemarks
        no_background_shoemarks = torch.cat(
            [no_background_shoemarks, generated_shoemarks]
        )
        combined_indices = torch.cat([no_background_indices, no_shoemark_indices])

    else:
        combined_indices = no_background_indices

    # Floor images for shoemarks requiring them
    floor_images = floor_images[
        shoemark_type_mask != ShoemarkImpressionType.SHOEMARK_BACK
    ]

    # Determine which need background substitution
    include_background = no_background_shoemarks.std(dim=(1, 2, 3)) > 0.08

    # Randomly apply pre blend transforms
    pre_blend_transformed = torch.rand(no_background_shoemarks.shape[0]) > 0.5
    pre_blend_mask = torch.zeros(shoemarks.size(0), dtype=torch.bool, device=device)
    pre_blend_mask[combined_indices[pre_blend_transformed]] = True

    no_background_shoemarks[pre_blend_transformed] = (
        streaming_transform.pre_blend_transforms(
            no_background_shoemarks[pre_blend_transformed]
        )
    )

    # Convert some shoemarks to blue (enhanced blood)
    no_background_shoemarks[include_background] = (
        streaming_transform.shoemark_blue_shift(
            no_background_shoemarks[include_background]
        )
    )

    # Don't affine cropped shoemarks
    no_background_shoemarks[~pre_blend_transformed] = (
        streaming_transform.shoemark_affine(
            no_background_shoemarks[~pre_blend_transformed]
        )
    )

    synth_background_shoemarks = (
        no_background_shoemarks[include_background]
        * floor_images[: include_background.sum()]
    )

    # Build list of (index, shoemark) tuples
    result_pairs = []

    # Add background shoemarks
    for idx, shoemark in zip(
        background_indices.tolist(), background_shoemarks, strict=True
    ):
        result_pairs.append((idx, shoemark))

    # Add synthetic background shoemarks
    synth_bg_iter = iter(synth_background_shoemarks)
    no_bg_iter = iter(no_background_shoemarks[~include_background])

    for idx, needs_bg in zip(
        combined_indices.tolist(), include_background.tolist(), strict=True
    ):
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
