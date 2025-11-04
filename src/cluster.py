"""Cluster shoeprints using a pre-trained model."""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from tqdm import tqdm

from siamese.config import load_config
from siamese.datasets import IndividualDataset
from siamese.model import SharedSiamese
from siamese.streaming import AdaptiveNormalisation

shoeprint_path = Path(
    "/home/struan/Vault/University/Doctorate/Data/GAN/Partitioned Clustered/Shoeprints/"
)
checkpoint_path = Path(
    "/home/struan/Development/Doctorate/siamese/checkpoints/new_part_baseline/siamese_9500.tar"
)
output_path = Path(
    "/home/struan/Vault/University/Doctorate/Data/GAN/Partitioned Clustered/clustered"
)
output_path.mkdir(exist_ok=True)

config = (
    load_config("../config.toml")
    if len(sys.argv) < 2 or sys.argv[1] == "" or sys.argv[1] == "-i"
    else load_config(sys.argv[1])
)

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

shoeprint_model = SharedSiamese(
    embedding_size=config["hyperparameters"]["embedding_size"],
    pre_trained=config["training"]["pre_training"]["pre_trained"],
    refreeze=config["training"]["pre_training"]["refreeze"],
    permafrost=config["training"]["pre_training"]["permafrost"],
).to(device)

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])
shoeprint_adaptive_norm = AdaptiveNormalisation(
    imagenet_mean, imagenet_std, device=device, momentum=0.9
)

checkpoint = torch.load(checkpoint_path, map_location=device)
shoeprint_model.load_state_dict(checkpoint["shoeprint_model_state_dict"])
shoeprint_adaptive_norm.load_state_dict(
    checkpoint["shoeprint_adaptive_norm_state_dict"]
)

shoeprint_model.eval()

shoeprint_dataset = IndividualDataset(shoeprint_path)

shoeprint_files = []
shoeprint_vectors = []

print("Extracting features from shoeprints...")
with torch.no_grad():
    for shoeprint, shoeprint_file in tqdm(shoeprint_dataset):
        shoeprint_files.append(shoeprint_file)
        shoeprint_gpu = shoeprint.to(device)
        shoeprint_norm = shoeprint_adaptive_norm(shoeprint_gpu, update=False)

        shoeprint_vector = shoeprint_model(shoeprint_norm)
        shoeprint_vectors.append(shoeprint_vector)

shoeprint_vectors = torch.cat(shoeprint_vectors).cpu().numpy()

shoeprint_vectors_norm = normalize(shoeprint_vectors, norm="l2")

print("Performing DBSCAN clustering...")
dbscan = DBSCAN(
    eps=0.1,  # Maximum distance between samples (tune this!)
    min_samples=2,  # Minimum samples in a neighborhood (tune this!)
    metric="cosine",
    n_jobs=-1,
)
labels = dbscan.fit_predict(shoeprint_vectors_norm)

# Print clustering statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"\nClustering complete!")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Create output directories and copy files
print("\nOrganizing files into cluster directories...")
output_path.mkdir(exist_ok=True)

# Create a noise directory for outliers
if n_noise > 0:
    noise_dir = output_path / "cluster_noise"
    noise_dir.mkdir(exist_ok=True)

for cluster_id in tqdm(set(labels)):
    if cluster_id == -1:
        cluster_dir = output_path / "cluster_noise"
    else:
        cluster_dir = output_path / f"cluster_{cluster_id:03d}"

    cluster_dir.mkdir(exist_ok=True)

    # Get all files belonging to this cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    for idx in cluster_indices:
        src_file = shoeprint_files[idx]
        dst_file = cluster_dir / src_file.name
        shutil.copy2(src_file, dst_file)

    # Print cluster size
    if cluster_id != -1:
        print(f"Cluster {cluster_id}: {len(cluster_indices)} images")

print(f"\nClustered shoeprints saved to: {output_path}")
