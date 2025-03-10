# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from logging import Logger
import torch
from glob import glob
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
from utils.fourier import ifft, fft
from utils.transforms import apply_mask
import utils.subsample as subsample

class CreateDataset:
    def __init__(
        self,
        data_path: str,
        train: bool,
        crop_size: int = 320,
        accelerations: list[int] = [8],
        single_coil: bool = False,
        image_domain: bool = False,
        standardize: bool = True,
        **kwargs,
    ):
        """
        Initialize the CreateDataset instance.
        """
        self.data_path = data_path
        self.file_list = glob(f"{data_path}/*.pt")
        self.train = train
        self.crop_size = crop_size
        self.accelerations = accelerations
        self.single_coil = single_coil
        self.image_domain = image_domain
        self.standardize = standardize

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.file_list)

    @staticmethod 
    def __load_tensor(path):
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
        self.data = self.__load_tensor(self.file_list[idx])
        if isinstance(self.data, dict):
            self.x = self.data["kspace"]
        else: 
            self.x = self.data
        self.y = self.data["reconstruction_rss"][None, ...]
        
        self.x = self.x.cfloat()
        
        if torch.isnan(self.x).any():
            self.x[torch.isnan(self.x)] = 0
        
        if self.single_coil:
            self.x = self.x.unsqueeze(0)
                
        if self.x.shape[-2:-1] != [320, 320]:
            self.reshape()
                            
        if self.train:         
            CreateDataset.undersample(self, center_fractions=[0.04], accelerations=self.accelerations) 
        else:
            CreateDataset.undersample(self, center_fractions=[0.04], accelerations=[self.accelerations[-1]])
        
        kspace_undersampled = self.x
        if self.image_domain:
            self.x = ifft(self.x)
        self.x, x_std, x_mean = CreateDataset.scaling(self.x)
        self.y, y_std, y_mean = CreateDataset.scaling(self.y)
        
        self.y = (self.y - self.y.min()) / (self.y.max() - self.y.min())
        
        scales = {"mean": x_mean, "std": x_std}

        return self.x, self.y, kspace_undersampled, scales
    
    def reshape(self):

        image = ifft(self.x)
        image_real = image.real
        image_imag = image.imag

        image_real = T.CenterCrop(self.crop_size)(image_real)
        image_imag = T.CenterCrop(self.crop_size)(image_imag)

        self.x = fft(image_real + 1j * image_imag)


    @staticmethod
    def scaling(data) -> torch.tensor:
        """
        Perform scaling on the data.
        """
        x_std = data.std()
        x_mean = data.mean()
        data = (data - x_mean) / (x_std + 1e-13)
            
        return data, x_std, x_mean

    @staticmethod
    def normalization(input):
        """
        Normalize the input data.
        """
        return (input - input.min()) / (input.max() - input.min())
    
    def undersample(self, center_fractions: list[float], accelerations: list[int]) -> None:
        """
        Undersample the input data.
        """
        mask_func = subsample.RandomMaskFunc(
            center_fractions=center_fractions,
            accelerations=accelerations,
            allow_any_combination=False,
        )
        
        # Change data dimensions for undersampling: last dimension is 2 (real and imaginary)
        self.x = torch.stack([self.x.real, self.x.imag], dim=-1)
        
        kspace_masked = apply_mask(self.x, mask_func)
        
        self.x = kspace_masked[0]
        
        # Change data dimensions back to original shape
        self.x = self.x[..., 0] + 1j * self.x[..., 1]

        
def get_loaders(
    save_path: str,
    train_path: str = None,
    val_path: str | dict[str] = None,
    csv_path: str = None,
    train_idx: list = None,
    val_idx: list = None,
    augmentation: str = None,
    transform: str = None,
    dim: int = 2,
    batch_size: int = 16,
    num_workers: int = 14,
    pin_memory: bool = True,
    log: Logger = None,
    single_coil: bool = False,
    image_domain: bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Get a dataloader for training and validation.

    Parameters:
        save_path (str): Path to save the indices.
        train_path (str): Path to the training data. Defaults to None.
        val_path (str | dict[str]): Path or dictionary of paths to the validation data. Defaults to an empty dictionary. Defaults to None.
        csv_path (str): Path to the csv file containing the data paths and unique ids. Defaults to None.
        train_idx (list): List of indices for training. Defaults to None.
        val_idx (list): List of indices for validation. Defaults to None.
        augmentation (str): Type of data augmentation. Defaults to None.
        transform (str): Type of data transformation. Defaults to None.
        dim (int): Dimensionality of the data. Defaults to 2.
        batch_size (int): Number of samples in each mini-batch. Defaults to 16.
        num_workers (int): Number of workers for data loading. Defaults to 5.
        pin_memory (bool): Whether to use pinned memory for faster data transfer. Defaults to True.
        log (Logger): Logger object for logging. Defaults to None.
        single_coil (bool): Whether to use single-coil data. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader, int]: Tuple containing training loader, validation loader, and total dataset size.
    """

    assert (train_path is not None) or (
        csv_path is not None
    ), "Either train_path or csv_path must be provided!"

    # Define dataset parameters
    dataset_params = {
        "augmentation": augmentation,
        "dim": dim,
        "transform": transform,
        "single_coil": single_coil,
        "image_domain": image_domain,
    }

    DatasetClass = CreateDataset

    if csv_path:
        csv = pd.read_csv(csv_path)
        train_path = csv.iloc[train_idx].reset_index()
        val_path = csv.iloc[val_idx].reset_index()
        if save_path:
            train_path.to_csv(f"{save_path}/train.csv", index=False)
            val_path.to_csv(f"{save_path}/val.csv", index=False)

    # Create train and validation datasets
    train_ds = DatasetClass(**dataset_params, data_path=train_path, train=True)
    val_ds = DatasetClass(**dataset_params, data_path=val_path, train=False)

    # Create DataLoader instances
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": 2
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_params)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_params)

    if log is not None:
        log.info(
            f"Found {len(train_ds)} examples in the training-set; {len(val_ds)} examples in the validation-set ..."
        )

    return train_loader, val_loader, (len(train_ds) + len(val_ds))

def get_test_loader(
    data_path: str,
    num_workers: int,
    batch_size: int = 16,
    single_coil: bool = False,
    image_domain: bool = False,
    standardize: bool = True,
):
    """
    Get a dataloader for testing.

    Parameters:
        data_path (str): Path to the test data.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Number of samples in each mini-batch. Defaults to 16.
        single_coil (bool): Whether to use single-coil data. Defaults to False.

    Returns:
        DataLoader: DataLoader instance for testing.
    """
    DatasetClass = CreateDataset

    test_ds = DatasetClass(data_path=data_path, train=False, single_coil=single_coil, image_domain=image_domain, standardize=standardize)

    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2
    }

    test_loader = DataLoader(test_ds, shuffle=False, **loader_params)

    return test_loader