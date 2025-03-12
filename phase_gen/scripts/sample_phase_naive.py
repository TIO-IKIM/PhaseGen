# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import os
from tqdm import tqdm
from utils.fourier import ifft
from glob import glob
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class CreateDataset:
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_path = self.data_path[idx]

        assert data_path.endswith(".pt") or data_path.endswith(
            ".npy"
        ), "Data should be in .pt or .np format"

        if data_path.endswith(".pt"):
            self.x = torch.load(data_path, weights_only=False).cfloat()
        elif data_path.endswith(".npy"):
            self.x = torch.tensor(np.load(data_path)).cfloat()
            data_path = data_path.replace(".npy", ".pt")

        self.x = ifft(self.x)

        return self.x, data_path


def get_data_list(data_path: str, save_path: str) -> list:
    """
    Retrieves a list of data files to process.

    Args:
        data_path (str): Path to the data directory.
        save_path (str): Path to the save directory.

    Returns:
        list: List of data file paths.
    """
    data_list = glob(f"{data_path}/*")
    existing_files = set(os.path.basename(f) for f in glob(f"{save_path}/*"))
    data_list = [f for f in data_list if os.path.basename(f) not in existing_files]

    return data_list


def get_data_loader(data_path: str, batch_size: int) -> DataLoader:
    """
    Creates a data loader for the given data list.

    Args:
        data_list (list): List of data file paths.
        batch_size (int): Batch size for the data loader.

    Returns:
        DataLoader: Data loader for the given data list.
    """
    test_ds = CreateDataset(data_path=data_path)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
    )

    return test_loader


def main(data_path: str, save_path: str):
    """
    Main function to process data and save phase images.

    Args:
        data_path (str): Path to the data directory.
        save_path (str): Path to the save directory.
    """
    data_list = get_data_list(data_path, save_path)

    data_loader = get_data_loader(data_list, 1)

    loop = tqdm(data_loader, desc="Processing batches")

    for i, (x, path) in enumerate(loop):
        for n in range(x.shape[0]):
            magnitude = x[n].squeeze().abs().cpu().numpy()

            # Normalize magnitude to [0, 1]
            magnitude_normalized = (magnitude - magnitude.min()) / (
                magnitude.max() - magnitude.min()
            )

            # Create a fake phase image
            rows, cols = magnitude.shape
            nx, ny = np.meshgrid(
                np.linspace(0, 2 * np.pi, cols), np.linspace(0, 2 * np.pi, rows)
            )
            phase_pattern = np.sin(nx) + np.cos(ny)  # Sinusoidal pattern

            # Modulate with magnitude and add randomness
            random_variation = np.random.randn(*magnitude.shape) * 0.05
            phase = phase_pattern * magnitude_normalized + random_variation

            fake_mri = magnitude * np.exp(1j * phase)

            plt.imsave(Path(save_path, Path(path[n]).stem + ".png"), np.angle(fake_mri))


if __name__ == "__main__":
    data_path = "/Path/to/data"
    save_path = "Path/to/save"

    main(data_path, save_path)
