# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from tqdm import tqdm
from utils.fourier import ifft, fft
from glob import glob
from torch.utils.data import DataLoader
import argparse
import numpy as np
import nibabel as nib
from utils.IKIMLogger import IKIMLogger

argparser = argparse.ArgumentParser(
    prog="sample_phase.py",
    description="Sample phase images from the given data directory using the given model.",
)
argparser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=True,
    help="Path to the model directory",
)
argparser.add_argument(
    "-i",
    "--data_path",
    type=str,
    required=True,
    help="Path to the data directory. Data needs to be in .pt or .npy format.",
)
argparser.add_argument(
    "-o",
    "--save_path",
    type=str,
    required=True,
    help="Path to the save directory.",
)


class CreateDataset:
    """
    Dataset class for loading data from .pt or .npy files.

    Attributes:
        data_path (list): List of data file paths.
    """

    def __init__(self, data_path: str, kspace: bool = False):
        self.data_path = data_path
        self.kspace = kspace

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        """
        Retrieves the data at the given index.

        Args:
            idx (int): Index of the data file.

        Returns:
            torch.Tensor: Loaded data tensor.
        """

        data_path = self.data_path[idx]

        if data_path.endswith(".pt"):
            self.x = torch.load(data_path, weights_only=False)
        elif data_path.endswith(".npy"):
            self.x = torch.tensor(np.load(data_path))
            data_path = data_path.replace(".npy", ".pt")
        elif data_path.endswith(".nii.gz"):
            self.x = torch.tensor(nib.load(data_path).get_fdata())
            self.x = torch.permute(self.x, (2, 0, 1))
            data_path = data_path.replace(".nii.gz", ".pt")

        if self.kspace:
            self.x = ifft(self.x).abs()

        if self.x.ndim == 2:
            self.x = self.x[None, ...]

        return self.x, data_path


class Sampler:
    """
    Sampler class for processing data and saving phase images.

    Attributes:
        data_path (str): Path to the data directory.
        save_path (str): Path to the save directory.
        model_path (str): Path to the model directory.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for the data loader.
        data_list (list): List of data file paths to process.
        model (torch.nn.Module): Loaded model.
        data_loader (DataLoader): Data loader for the given data list.
    """

    def __init__(
        self, data_path, save_path, model_path, device, batch_size, kspace: bool = False
    ):
        """
        Initializes the Sampler with the given parameters.

        Args:
            data_path (str): Path to the data directory.
            save_path (str): Path to the save directory.
            device (torch.device): Device to run the model on.
            batch_size (int): Batch size for the data loader.
        """
        self.data_path = data_path
        self.save_path = save_path
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.kspace = kspace
        self._get_data_list()
        self._load_model()
        self._get_data_loader()

    def _get_data_list(self):
        """
        Retrieves a list of data files to process and filters out existing files in the save directory.
        """
        if os.path.isfile(self.data_path):
            data_list = [self.data_path]
        else:
            data_list = []
            for ext in ["**/*.pt", "**/*.npy", "**/*.nii.gz"]:
                data_list.extend(glob(f"{self.data_path}/{ext}", recursive=True))
        existing_files = set(os.path.basename(f) for f in glob(f"{self.save_path}/*"))
        self.data_list = [
            f for f in data_list if os.path.basename(f) not in existing_files
        ]

    def _load_model(self):
        """
        Loads the model from the model path and sets it to evaluation mode.
        """
        self.model = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        self.model.to(self.device)

    def _get_data_loader(self):
        """
        Creates a data loader for the given data list.
        """
        test_ds = CreateDataset(data_path=self.data_list, kspace=self.kspace)

        self.data_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
        )

    def __call__(self):
        """
        Processes the data and saves the phase images.
        """
        logger.info(
            f"Sampling {len(self.data_list)} files ({len(self.data_loader)} batches)."
        )

        for x, path in tqdm(self.data_loader, desc="Processing batches"):
            x = x.to(self.device, memory_format=torch.channels_last)
            with torch.no_grad():
                if x.shape[1] > 1:
                    data_with_phase = torch.cat(
                        [
                            self.model.sample(x[:, i, ...].unsqueeze(1)).unsqueeze(1)
                            for i in range(x.shape[1])
                        ],
                        dim=1,
                    )
                else:
                    data_with_phase = self.model.sample(x)

                data_with_phase[torch.isnan(data_with_phase)] = 0

            data_with_phase = (
                fft(data_with_phase.squeeze())
                if self.kspace
                else data_with_phase.squeeze()
            )
            data_with_phase = data_with_phase.detach().cpu()

            if data_with_phase.ndim == 3:
                torch.save(
                    data_with_phase.clone(),
                    os.path.join(self.save_path, os.path.basename(path[0])),
                )
                logger.info(f"Saved {os.path.basename(path[0])} to {self.save_path}")
            else:
                for j in range(x.shape[0]):
                    torch.save(
                        data_with_phase[j].clone(),
                        os.path.join(self.save_path, os.path.basename(path[j])),
                    )
                    logger.info(
                        f"Saved {os.path.basename(path[j])} to {self.save_path}"
                    )
        logger.info("---- Finished sampling ----.")


if __name__ == "__main__":
    args = argparser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    save_path = args.save_path

    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir="logs",
        comment="phase_sampling",
    )
    logger = ikim_logger.create_logger()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampler = Sampler(data_path, save_path, model_path, device, 1, kspace=False)
    sampler()
