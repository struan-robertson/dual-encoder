"""Validation functions."""

import sys
from pathlib import Path

import torch
import torchvision
from tqdm import trange

from siamese.config import load_config
from siamese.streaming import StreamingDataset, StreamingTransforms, augment_batch


# Remember to run with a batch size of 1 so that each image augmentation is unique
def generate_val_images(save_dir: Path, total_per_shoeprint: int = 10):
    """Generate a set of augmented validation images."""
    config = (
        load_config("config.toml")
        if len(sys.argv) < 2 or sys.argv[1] == "" or sys.argv[1] == "-i"
        else load_config(sys.argv[1])
    )

    if config["gan"]["enabled"]:
        # Only import the generator backend when it is enabled
        if config["gan"]["backend"] == "unsb":
            sys.path.insert(0, str(config["gan"]["config"].parent))
            from unsb_handler import GeneratorHandler, load_unsb_opt  # noqa: PLC0415

            gan_opt = load_unsb_opt(config["gan"]["config"])
            device = torch.device(
                f"cuda:{gan_opt.gpu_ids[0]}"
                if torch.cuda.is_available() and gan_opt.gpu_ids
                else "cpu"
            )
            generator_handler = GeneratorHandler(gan_opt, device)
        else:
            from one_to_many_gan import GeneratorHandler, load_gan_config  # noqa: PLC0415

            gan_config = load_gan_config(config["gan"]["config"])
            device = torch.device(
                f"cuda:{gan_config['training']['gpu_number']}"
                if torch.cuda.is_available()
                else "cpu"
            )
            generator_handler = GeneratorHandler(gan_config, device)
    else:
        device = torch.device(
            f"cuda:{config['training']['gpu_number']}"
            if torch.cuda.is_available()
            else "cpu"
        )
        generator_handler = None

    val_dataset = StreamingDataset(
        Path("/home/struan/Development/Doctorate/siamese/data/Shoeprints/val"),
        Path("/home/struan/Development/Doctorate/siamese/data/tmp"),
        Path("/home/struan/Development/Doctorate/siamese/data/flooring"),
        config["data"]["image_size"],
        config["data"]["streaming"]["min_floor_roi_height"],
        synthetic_ratio=0,
        labelled=True,
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
        config["augmentations"], size=config["data"]["image_size"]
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
                shoeprints, shoemarks = augment_batch(
                    shoeprint_batch.to(device),
                    shoeprint_gen_batch.to(device),
                    floor_image_batch.to(device),
                    shoemark_batch.to(device),
                    shoemark_type_mask_batch.to(device),
                    streaming_transform=streaming_transform,
                    device=device,
                    generator_handler=generator_handler,
                    difficulty=1.0,
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
