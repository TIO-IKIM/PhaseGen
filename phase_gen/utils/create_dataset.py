# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from logging import Logger
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils.fourier import ifft
from torchvision import transforms


class CreateDataset:

    def __init__(
        self,
        data_csv,
        **kwargs,
    ):
        """
        Initialize the CreateDataset instance.
        """
        self.y_path = data_csv["file"]
        self.crop_size = 256

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.y_path)

    def __load_tensor(self, path):
        """
        Load a tensor from a given path.
        """
        return torch.load(path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input tensor (self.x) and the target tensor (self.y).
        """
        self.x = self.__load_tensor(self.y_path[idx])
        self.y = self.__load_tensor(self.y_path[idx])

        if self.x.shape != [256, 256]:
            self.x = self.reshape(self.x)
            self.y = self.reshape(self.y)

        self.x = (self.x).cfloat()
        self.y = (self.y).cfloat()
        self.x = self.x[None, ...]
        self.y = self.y[None, ...]

        CreateDataset.standardization(self)

        self.x = ifft(self.x)
        self.y = ifft(self.y)

        return self.x, self.y

    def standardization(self) -> None:
        """
        Perform standardization on the data.
        """
        self.x = (self.x - self.x.mean()) / self.x.std()
        self.y = (self.x - self.y.mean()) / self.y.std()

    def reshape(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input complex tensor by applying a center crop transformation.
        Args:
            data (torch.Tensor): A complex tensor with real and imaginary parts.
        Returns:
            torch.Tensor: The reshaped complex tensor after applying center crop.
        """

        image_real = data.real
        image_imag = data.imag

        image_real = transforms.CenterCrop(self.crop_size)(image_real)
        image_imag = transforms.CenterCrop(self.crop_size)(image_imag)

        data = image_real + 1j * image_imag

        return data


def get_loaders(
    csv_path: str,
    train_idx: list,
    val_idx: list,
    batch_size: int = 16,
    num_workers: int = 5,
    pin_memory: bool = True,
    log: Logger = None,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Creates and returns DataLoader instances for training and validation datasets.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        train_idx (list): List of indices for the training set.
        val_idx (list): List of indices for the validation set.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 5.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
        log (Logger, optional): Logger instance for logging information. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader, int]: A tuple containing the training DataLoader, validation DataLoader, and the total number of examples.
    """

    DatasetClass = CreateDataset

    csv = pd.read_csv(csv_path)
    train_files = csv.iloc[train_idx].reset_index()
    val_files = csv.iloc[val_idx].reset_index()

    # Create train and validation datasets
    train_ds = DatasetClass(data_csv=train_files)
    val_ds = DatasetClass(data_csv=val_files)

    # Create DataLoader instances
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": True,
    }

    train_loader = DataLoader(train_ds, **loader_params)
    val_loader = DataLoader(val_ds, **loader_params)

    if log is not None:
        log.info(
            f"Found {len(train_ds)} examples in the training-set; {len(val_ds)} examples in the validation-set ..."
        )

    return train_loader, val_loader, (len(train_ds) + len(val_ds))


def get_test_loader(
    test_path: str,
    transform: str = None,
    dim: int = 2,
    filenum: int = 8,
    num_workers: int = 5,
    pin_memory: bool = True,
) -> tuple[DataLoader, int]:
    """
    Get a data loader for testing.

    Parameters:
        test_path (str): Path to the test data.
        transform (str): Type of data transformation. Defaults to None.
        dim (int): Dimensionality of the data. Defaults to 2.
        filenum (int): Number of samples in each mini-batch. Defaults to 8.
        num_workers (int): Number of workers for data loading. Defaults to 5.
        pin_memory (bool): Whether to use pinned memory for faster data transfer. Defaults to True.

    Returns:
        tuple[DataLoader, int]: Tuple containing the test loader and the total number of examples in the dataset.
    """

    # Define dataset parameters
    dataset_params = {
        "dim": dim,
        "transform": transform,
    }

    test_ds = CreateDataset(**dataset_params, data_path=test_path)

    test_loader = DataLoader(
        test_ds,
        batch_size=filenum,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader, len(test_ds)


class Folds:
    """
    Folds class for creating cross-validation folds from a CSV file.

    Attributes:
        folds (list): List of train/test indices for each fold.
    """

    def __init__(self, csv_path, n_folds):
        """
        Initializes the Folds class with the given CSV path and number of folds.

        Args:
            csv_path (str): Path to the CSV file containing the data.
            n_folds (int): Number of folds for cross-validation.
        """
        df = pd.read_csv(csv_path)
        x = df["file"]
        y = df["file"]
        group = df["id"]

        le = LabelEncoder()
        group = le.fit_transform(group)

        group_kfold = GroupKFold(n_splits=n_folds)

        self.folds = list(group_kfold.split(x, y, group))

    def __len__(self):
        """
        Returns the number of folds.

        Returns:
            int: Number of folds.
        """
        return len(self.folds)

    def __getitem__(self, idx):
        """
        Retrieves the train/test indices for the given fold index.

        Args:
            idx (int): Index of the fold.

        Returns:
            tuple: Train/test indices for the given fold.
        """
        return self.folds[idx]
