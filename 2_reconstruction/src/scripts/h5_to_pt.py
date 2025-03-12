# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from pathlib import Path
import os, sys
import h5py
import torch
import torchvision.transforms as T
from tqdm import tqdm
from glob import glob
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.fourier import ifft, fft

parser = argparse.ArgumentParser(prog="Prepare FastMRI_Prostate data")

parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    help="Path to input files",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="Path to save files",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BrainSave:
    """Class to save brain MRI data in h5-format as .pt files.

    Args:
        fastMRI_path (str): Path to the directory containing .h5 files.
        save_path (str): Path to the directory where processed files will be saved.
    """

    def __init__(self, fastMRI_path: str, save_path: str) -> None:
        """Initializes BrainSave with paths for input and output.

        Args:
            fastMRI_path (str): Path to the directory containing .h5 files.
            save_path (str): Path to the directory where processed files will be saved.
        """
        self.fastMRI_path = fastMRI_path
        self.save_path = Path(save_path)
        if self.save_path.exists() == False:
            Path.mkdir(self.save_path)

    def get_file_paths(self):
        """Gets the file paths of all .h5 files in the fastMRI_path directory.

        Sets:
            self.file_paths (list): List of file paths to .h5 files.
        """
        self.file_paths = glob(self.fastMRI_path + "/*.h5")
        existing_files = glob(str(self.save_path) + "/*.pt")
        existing_files = [Path(file).name.split("_")[0] for file in existing_files]

        self.file_paths = [
            file for file in self.file_paths if Path(file).stem not in existing_files
        ]

    @staticmethod
    def load_h5(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads k-space and reconstruction data from an .h5 file.

        Args:
            file_path (str): Path to the .h5 file.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The k-space data.
                - torch.Tensor: The reconstruction data.
        """
        hf = h5py.File(file_path, "r")
        volume_kspace = hf["kspace"][()]
        try:
            volume_recon = hf["reconstruction_rss"][()]
        except:
            volume_recon = None
            return torch.tensor(volume_kspace), None

        return torch.tensor(volume_kspace), torch.tensor(volume_recon)

    def crop_center(self, kspace: torch.Tensor, crop_size: tuple) -> torch.Tensor:
        """Crops the center of the k-space data, by transforming it first into image domain.
        After cropping, the data is transformed back into the k-space.

        Args:
            kspace (torch.Tensor): The k-space data.
            crop_size (tuple): The size to which the center of the k-space data should be cropped.

        Returns:
            torch.Tensor: The cropped k-space data.
        """

        image = ifft(kspace)
        image_real = image.real
        image_imag = image.imag

        image_real = T.CenterCrop(crop_size)(image_real)
        image_imag = T.CenterCrop(crop_size)(image_imag)

        kspace = fft(image_real + 1j * image_imag)

        return kspace

    @staticmethod
    def save_slice(volume_kspace, volume_recon, file_path):
        """Saves slices of k-space and reconstruction data in one file as dictionary
        with the keys 'kspace' and 'reconstruction_rss'.

        Args:
            volume_kspace (torch.Tensor): The k-space data.
            volume_recon (torch.Tensor): The reconstruction data.
            file_path (str): The path where the slices will be saved.

        Raises:
            AssertionError: If the shapes of k-space and reconstruction data do not match.
        """
        if volume_recon is None:
            for slice in range(volume_kspace.shape[0]):
                torch.save(
                    {
                        "kspace": torch.clone(volume_kspace[slice].detach()),
                        "reconstruction_rss": None,
                    },
                    str(file_path) + f"_slice_{slice}.pt",
                )
        else:
            assert volume_kspace.shape[-2:] == volume_recon.shape[-2:], "Shape mismatch"

            for slice in range(volume_kspace.shape[0]):
                torch.save(
                    {
                        "kspace": torch.clone(volume_kspace[slice].detach()),
                        "reconstruction_rss": torch.clone(volume_recon[slice].detach()),
                    },
                    str(file_path) + f"_slice_{slice}.pt",
                )

    def __call__(self, *args, **kwds):
        logging.info(f"Loading files from {self.fastMRI_path} ...")
        self.get_file_paths()
        logging.info(f"Found {len(self.file_paths)} volumes.")

        logging.info(f"Saving files to {self.save_path} ...")

        for file_path in tqdm(self.file_paths):
            save_path = Path(self.save_path, Path(file_path).stem)

            volume_kspace, volume_recon = self.load_h5(file_path)

            if volume_recon is not None:
                crop_size = volume_recon.shape[-2], volume_recon.shape[-1]
                volume_kspace = self.crop_center(volume_kspace, crop_size)

            self.save_slice(volume_kspace, volume_recon, save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    brain_save = BrainSave(args.input_path, args.output_path)
    brain_save()
