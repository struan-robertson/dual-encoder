"""GPU transformations for streaming data."""

from typing import TYPE_CHECKING

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F

from siamese.streaming import ShoemarkImpressionType

if TYPE_CHECKING:
    from one_to_many_gan import GeneratorHandler

    from siamese.config import Augmentations


class RandomCropAndPad:
    """Randomly crop a batch of tensors and then scale & pad back to original shape."""

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


def _build_affine(config) -> transforms.Transform:
    """Build a flip + affine transform from an augmentation config table."""
    if not config["enabled"]:
        return transforms.Identity()

    steps: list[transforms.Transform] = []
    if config["flip"]:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(
        transforms.RandomAffine(
            degrees=config["degrees"],  # pyright: ignore [reportArgumentType]
            translate=tuple(config["translate"]),
            scale=tuple(config["scale"]),
            fill=config["fill"],
            shear=[config["shear"]] * 4,
        )
    )
    return transforms.Compose(steps)


class StreamingTransforms:
    """Transforms used in the streaming dataloader.

    Each augmentation is built from its table in the [augmentations] config
    section; disabled augmentations become identity transforms so call sites
    never need to branch.
    """

    def __init__(self, aug_config: "Augmentations", size: tuple[int, int]):
        self.shoeprint_affine = _build_affine(aug_config["shoeprint_affine"])
        self.shoemark_affine = _build_affine(aug_config["shoemark_affine"])
        self.shoemark_back_affine = _build_affine(aug_config["shoemark_back_affine"])

        # Whether non-dust no-background shoemarks are blended onto floor images
        self.background_substitution = aug_config["background_substitution"]["enabled"]

        self.invert = (
            transforms.RandomInvert()
            if aug_config["dust_invert"]["enabled"]
            else transforms.Identity()
        )

        blue_shift = aug_config["blue_shift"]
        if blue_shift["enabled"]:
            max_shift = blue_shift["max_shift"]

            def shoemark_blue_shift(shoemark: torch.Tensor):
                shift_amount = (
                    torch.rand(shoemark.shape[0], device=shoemark.device) * max_shift
                )
                shoemark[:, 2, :, :].add_(shift_amount[:, None, None])
                return shoemark

            self.shoemark_blue_shift = transforms.RandomApply(
                [
                    shoemark_blue_shift,
                    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                ],
                p=blue_shift["p"],
            )
        else:
            self.shoemark_blue_shift = transforms.Identity()

        erasing = aug_config["pre_blend_erasing"]
        # Probability of applying pre blend transforms, used in shoemark_pipeline
        self.pre_blend_p = erasing["p"] if erasing["enabled"] else 0.0
        self.pre_blend_transforms = (
            transforms.RandomErasing(
                p=1,
                scale=(erasing["min_scale"], erasing["max_scale"]),
                ratio=(0.3, 3.33),
                value=erasing["fill"],
            )
            if erasing["enabled"]
            else transforms.Identity()
        )

        crop = aug_config["post_blend_crop"]
        self.post_blend_transform = (
            transforms.RandomApply(
                transforms=[
                    RandomCropAndPad(
                        fill=crop["fill"], min_edge=crop["min_edge"], size=size
                    )
                ],
                p=crop["p"],
            )
            if crop["enabled"]
            else transforms.Identity()
        )

        photometric = aug_config["photometric"]
        if photometric["enabled"]:
            self.photometric_transforms = transforms.Compose(
                [
                    transforms.RandomChoice(
                        [
                            transforms.GaussianBlur(
                                kernel_size=tuple(photometric["blur_kernel"]),
                                sigma=tuple(photometric["blur_sigma"]),
                            ),
                            transforms.RandomAdjustSharpness(
                                sharpness_factor=photometric["sharpness_factor"]
                            ),
                        ]
                    ),
                    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=photometric["brightness"],
                                hue=photometric["hue"],
                            )
                        ],
                        p=photometric["jitter_p"],
                    ),
                    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                ]
            )
        else:
            self.photometric_transforms = transforms.Identity()


def shoemark_pipeline(
    shoeprints: torch.Tensor,
    floor_images: torch.Tensor,
    shoemarks: torch.Tensor,
    shoemark_type_mask: torch.Tensor,
    streaming_transform: "StreamingTransforms",
    device,
    generator_handler: "GeneratorHandler | None" = None,
    difficulty: float = 1.0,
    synthetic_pool: bool = False,
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
        if synthetic_pool:
            # Pre-generated marks streamed from disk by the dataset in [0, 1];
            # map back to the generator's tanh range so downstream treatment
            # matches in-loop generation exactly
            generated_shoemarks = shoemarks[no_shoemark_mask] * 2.0 - 1.0
        elif generator_handler is not None:
            # Generate shoemarks using shoeprints
            generated_shoemarks = generator_handler.generate(
                F.rgb_to_grayscale(no_shoemark_shoeprints),
                difficulty=difficulty,
                normalised=False,
            )
        else:
            # No GAN: use shoeprints directly, converted to grayscale to match
            # the format the GAN would produce
            generated_shoemarks = F.rgb_to_grayscale(no_shoemark_shoeprints)

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

    # Low-texture shoemarks are treated as dust marks
    is_dust = no_background_shoemarks.std(dim=(1, 2, 3)) <= 0.08

    # Non-dust shoemarks are blended onto floor images (unless disabled)
    if streaming_transform.background_substitution:
        include_background = ~is_dust
    else:
        include_background = torch.zeros_like(is_dust)

    # Randomly invert dust shoemarks
    no_background_shoemarks[is_dust] = streaming_transform.invert(
        no_background_shoemarks[is_dust]
    )

    # Randomly apply pre blend transforms to dust shoemarks
    pre_blend_transformed = (
        torch.rand(no_background_shoemarks.shape[0], device=device)
        < streaming_transform.pre_blend_p
    ) & is_dust
    pre_blend_mask = torch.zeros(shoemarks.size(0), dtype=torch.bool, device=device)
    pre_blend_mask[combined_indices[pre_blend_transformed]] = True

    no_background_shoemarks[pre_blend_transformed] = (
        streaming_transform.pre_blend_transforms(
            no_background_shoemarks[pre_blend_transformed]
        )
    )

    # Convert some non-dust shoemarks to blue (enhanced blood)
    no_background_shoemarks[~is_dust] = streaming_transform.shoemark_blue_shift(
        no_background_shoemarks[~is_dust]
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

    # We need to know which shoemarks have been pre-blend transformed as
    # pre/post blending transformations are mutually exclusive

    result_indices = torch.tensor([idx for idx, _ in result_pairs], device=device)
    sorted_pre_blended = pre_blend_mask[result_indices]

    return torch.stack([shoemark for _, shoemark in result_pairs]), sorted_pre_blended


def augment_batch(
    shoeprints: torch.Tensor,
    shoeprints_gen: torch.Tensor,
    floor_images: torch.Tensor,
    shoemarks: torch.Tensor,
    shoemark_type_mask: torch.Tensor,
    streaming_transform: "StreamingTransforms",
    device,
    generator_handler: "GeneratorHandler | None" = None,
    difficulty: float = 1.0,
    synthetic_pool: bool = False,
):
    """Run the full augmentation pipeline on a batch.

    Creates shoemarks with shoemark_pipeline then applies the affine,
    post blend, and photometric transforms. Returns the augmented
    (shoeprints, shoemarks) pair.
    """
    shoemarks, pre_blended_mask = shoemark_pipeline(
        shoeprints_gen,
        floor_images,
        shoemarks,
        shoemark_type_mask,
        streaming_transform=streaming_transform,
        device=device,
        generator_handler=generator_handler,
        difficulty=difficulty,
        synthetic_pool=synthetic_pool,
    )

    shoeprints = streaming_transform.shoeprint_affine(shoeprints)

    background_mask = shoemark_type_mask == ShoemarkImpressionType.SHOEMARK_BACK

    # Apply post blend transforms to shoemarks that did not feature pre blend
    # transforms (they are mutually exclusive) or a pre-existing background
    shoemarks[~pre_blended_mask & ~background_mask] = (
        streaming_transform.post_blend_transform(
            shoemarks[~pre_blended_mask & ~background_mask]
        )
    )

    # Apply photometric transforms to shoemarks without a pre-existing background
    shoemarks[~background_mask] = streaming_transform.photometric_transforms(
        shoemarks[~background_mask]
    )

    # Smaller affine with black fill for shoemarks with a pre-existing background
    shoemarks[background_mask] = streaming_transform.shoemark_back_affine(
        shoemarks[background_mask]
    )

    return shoeprints, shoemarks
