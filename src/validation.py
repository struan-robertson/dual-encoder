"""Validation functions."""

import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torchvision
from tqdm import trange

from siamese.config import load_config
from siamese.streaming import StreamingDataset, StreamingTransforms, augment_batch


# Remember to run with a batch size of 1 so that each image augmentation is unique
def generate_val_images(save_dir: Path, total_per_shoeprint: int = 10):
    """Generate a set of augmented validation images."""
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.toml")

    device = torch.device(
        f"cuda:{config.training.gpu_number}" if torch.cuda.is_available() else "cpu"
    )

    val_dataset = StreamingDataset(
        Path("/home/struan/Development/Doctorate/siamese/data/Shoeprints/val"),
        Path("/home/struan/Development/Doctorate/siamese/data/tmp"),
        Path("/home/struan/Development/Doctorate/siamese/data/flooring"),
        config.data.image_size,
        config.data.streaming.min_floor_roi_height,
        synthetic_ratio=0,
        labelled=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.hyperparameters.batch_size,
        shuffle=False,
        num_workers=0,  # All images are in RAM
        pin_memory=True,
        drop_last=False,
    )

    streaming_transform = StreamingTransforms(
        asdict(config.augmentations), size=config.data.image_size
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
