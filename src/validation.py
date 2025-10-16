"""Validation functions."""

import sys
from pathlib import Path

import torch
import torchvision
from one_to_many_gan import GeneratorHandler, load_gan_config
from tqdm import trange

from siamese.config import load_config
from siamese.streaming import (
    ShoemarkImpressionType,
    StreamingDataset,
    StreamingTransforms,
    shoemark_pipeline,
)


# Remember to run with a batch size of 1 so that each image augmentation is unique
def generate_val_images(save_dir: Path, total_per_shoeprint: int = 10):
    """Generate a set of augmented validation images."""
    config = (
        load_config("config.toml")
        if len(sys.argv) < 2 or sys.argv[1] == "" or sys.argv[1] == "-i"
        else load_config(sys.argv[1])
    )
    gan_config = load_gan_config(config["training"]["gan_config"])

    gan_device = torch.device(
        f"cuda:{gan_config['training']['gpu_number']}"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator_handler = GeneratorHandler(gan_config, gan_device)

    val_dataset = StreamingDataset(
        Path("/home/struan/Development/Doctorate/siamese/data/Shoeprints/val"),
        Path("/home/struan/Development/Doctorate/siamese/data/tmp"),
        Path("/home/struan/Development/Doctorate/siamese/data/flooring"),
        config["data"]["image_size"],
        config["data"]["streaming"]["min_floor_roi_height"],
        synthetic_ratio=0,
        val=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["hyperparameters"]["batch_size"],
        shuffle=False,
        num_workers=0,  # All images are in RAM
        pin_memory=True,
        drop_last=False,
    )

    streaming_transform = StreamingTransforms(
        fill=0.0, min_edge=225, size=config["data"]["image_size"]
    )

    shoeprint_dir = save_dir / "Shoeprints"
    shoemark_dir = save_dir / "Shoemarks"
    shoeprint_dir.mkdir(exist_ok=True)
    shoemark_dir.mkdir(exist_ok=True)

    for epoch in trange(total_per_shoeprint):
        for (
            shoeprint_batch,
            shoeprint_gen_batch,
            floor_image_batch,
            shoemark_batch,
            shoemark_type_mask_batch,
            shoeprint_classes,
        ) in val_loader:
            with torch.no_grad():
                shoeprints = shoeprint_batch.to(gan_device)
                shoeprints_gen = shoeprint_gen_batch.to(gan_device)
                floor_images = floor_image_batch.to(gan_device)
                shoemarks = shoemark_batch.to(gan_device)
                shoemark_type_mask = shoemark_type_mask_batch.to(gan_device)

                shoemarks, pre_blended_mask = shoemark_pipeline(
                    shoeprints_gen,
                    floor_images,
                    shoemarks,
                    shoemark_type_mask,
                    generator_handler,
                    difficulty=1.0,
                    streaming_transform=streaming_transform,
                    device=gan_device,
                )

                shoeprints = streaming_transform.shoeprint_affine(shoeprints)

                # Apply post blend transforms to shoemarks that did not feature pre
                # blend transforms or pre-existing background
                background_mask = (
                    shoemark_type_mask == ShoemarkImpressionType.SHOEMARK_BACK
                )
                shoemarks[~pre_blended_mask & ~background_mask] = (
                    streaming_transform.post_blend_transform(
                        shoemarks[~pre_blended_mask & ~background_mask]
                    )
                )

                # Apply photometric transforms to non back shoemarks
                shoemarks[~background_mask] = (
                    streaming_transform.photometric_transforms(
                        shoemarks[~background_mask]
                    )
                )

                # Smaller affine with black fill
                shoemarks[background_mask] = streaming_transform.shoemark_back_affine(
                    shoemarks[background_mask]
                )

                shoeprints = shoeprints.cpu()
                shoemarks = shoemarks.cpu()

                # Doesn'r really matter that we're overwriting
                for i, shoeprint in enumerate(shoeprints):
                    torchvision.utils.save_image(
                        shoeprint, shoeprint_dir / f"{shoeprint_classes[i]}.png"
                    )

                for i, shoemark in enumerate(shoemarks):
                    torchvision.utils.save_image(
                        shoemark, shoemark_dir / f"{shoeprint_classes[i]}_{epoch}.png"
                    )
