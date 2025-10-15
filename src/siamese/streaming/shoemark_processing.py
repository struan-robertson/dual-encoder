"""Processing of shoemarks."""

import torch
import torchvision.transforms.v2.functional as F
from one_to_many_gan import GeneratorHandler
from siamese.streaming import ShoemarkImpressionType, StreamingTransforms


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
        F.rgb_to_grayscale(no_shoemark_shoeprints),
        difficulty=difficulty,
        normalised=False,
    )
    b, _, h, w = generated_shoemarks.shape
    generated_shoemarks = generated_shoemarks.expand(b, 3, h, w)

    # Floor images for shoemarks requiring them
    floor_images = floor_images[
        shoemark_type_mask != ShoemarkImpressionType.SHOEMARK_BACK
    ]

    # Combine no_background and generated shoemarks
    no_background_shoemarks = torch.cat([no_background_shoemarks, generated_shoemarks])
    combined_indices = torch.cat([no_background_indices, no_shoemark_indices])

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
