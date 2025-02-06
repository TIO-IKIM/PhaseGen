import torch
import os
from tqdm import tqdm
from utils.fourier import ifft, fft
from glob import glob
from torch.utils.data import DataLoader
import argparse
import numpy as np

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
    "-o", "--save_path", type=str, required=True, help="Path to the save directory."
)


class CreateDataset:
    """
    Dataset class for loading data from .pt or .npy files.

    Attributes:
        data_path (list): List of data file paths.
    """

    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

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

        assert data_path.endswith(".pt") or data_path.endswith(
            ".npy"
        ), "Data should be in .pt or .npy format"

        if data_path.endswith(".pt"):
            self.x = torch.load(data_path, weights_only=False)
        elif data_path.endswith(".npy"):
            self.x = torch.tensor(np.load(data_path))
            data_path = data_path.replace(".npy", ".pt")
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

    def __init__(self, data_path, save_path, device, batch_size):
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
        self.__get_data_list()
        self.__load_model()
        self.__get_data_loader()

    def _get_data_list(self):
        """
        Retrieves a list of data files to process and filters out existing files in the save directory.
        """
        data_list = glob(f"{self.data_path}/*")
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
        self.model.eval()

    def _get_data_loader(self):
        """
        Creates a data loader for the given data list.
        """
        test_ds = CreateDataset(data_path=self.data_list)

        self.data_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True,
        )

    def __call__(self):
        """
        Processes the data and saves the phase images.
        """
        loop = tqdm(self.data_loader, desc="Processing batches")

        for i, (x, path) in enumerate(loop):
            x = x.to(self.device)
            with torch.no_grad():
                if x.shape[1] == 1:
                    data_with_phase = self.model.sample(x)
                else:
                    for i in range(x.shape[1]):
                        data_with_phase = self.model.sample(x[:, i, ...].unsqueeze(1))
                        if torch.isnan(data_with_phase).any():
                            data_with_phase[torch.isnan(data_with_phase)] = 0
                        if i == 0:
                            data_with_phase_stack = data_with_phase.unsqueeze(1)
                        else:
                            data_with_phase_stack = torch.cat(
                                (data_with_phase_stack, data_with_phase.unsqueeze(1)),
                                dim=1,
                            )
                    data_with_phase = data_with_phase_stack

            data_fft = fft(data_with_phase.squeeze().detach().cpu())
            torch.save(
                data_fft.clone(),
                os.path.join(self.save_path, os.path.basename(path[0])),
            )


if __name__ == "__main__":

    args = argparser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    save_path = args.save_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampler = Sampler(data_path, save_path, device, 64)
    sampler()
