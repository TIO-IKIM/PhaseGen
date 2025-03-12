# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import os
import logging
from tqdm import tqdm
from utils.fourier import ifft
from glob import glob
import concurrent.futures

logging.basicConfig(
    filename="./create_label.log",
    filemode="w+",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)


def load_data(filename: str, orig_path: str, label_path: str) -> tuple:
    """
    Loads the original data and corresponding label from the given paths.

    Args:
        filename (str): Name of the file to load.
        orig_path (str): Path to the original data.
        label_path (str): Path to the label data.

    Returns:
        tuple: A tuple containing the original data tensor and the label tensor.
    """
    orig_path = os.path.join(orig_path, filename)
    orig = torch.load(orig_path, weights_only=False)
    orig = ifft(orig)

    full_label_path = os.path.join(label_path, filename)
    if not os.path.exists(full_label_path):
        filename_new = "_".join(
            filename.split("_")[:-1] + ["staple"] + filename.split("_")[-1:]
        )
        full_label_path = os.path.join(label_path, filename_new)
    if not os.path.exists(full_label_path):
        filename_new = "_".join(
            filename.split("_")[:-1] + ["brainmask"] + filename.split("_")[-1:]
        )
        full_label_path = os.path.join(label_path, filename_new)
    if os.path.exists(full_label_path):
        label = torch.load(full_label_path, weights_only=False)
        label = ifft(label)
    else:
        logging.warning(f"Label not found for {filename}")
        label = None

    return orig, label


def create_mask(label: torch.Tensor) -> torch.Tensor:
    """
    Creates a mask from the given label.

    Args:
        label (torch.Tensor): Label tensor.

    Returns:
        torch.Tensor: Mask tensor.
    """
    mask = (label.abs() > 0.01).float()

    return mask


def create_label(orig: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Creates a label by applying the mask to the original data.

    Args:
        orig (torch.Tensor): Original data tensor.
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: Label tensor.
    """
    label = orig * mask
    return label


def save_label(masked_orig: torch.Tensor, save_path: str, filename: str) -> None:
    torch.save(masked_orig, f"{save_path}/{filename}")


def get_data_list(orig_path: str, save_path: str) -> list:
    """
    Generates a list of data files from the original path that do not already exist in the save path.
    Args:
        orig_path (str): The directory path where the original data files are located.
        save_path (str): The directory path where the processed or saved data files are located.
    Returns:
        list: A list of file paths from the original directory that are not present in the save directory.
    """

    orig_list = glob(f"{orig_path}/*.pt")
    existing_files = set(os.path.basename(f) for f in glob(f"{save_path}/*"))
    orig_list = [f for f in orig_list if os.path.basename(f) not in existing_files]

    return orig_list


def main(filename: str, orig_path: str, label_path: str, save_path: str) -> None:
    """
    Main function to process and create a labeled image.
    Args:
        filename (str): The name of the file to process.
        orig_path (str): The path to the original images.
        label_path (str): The path to the label images.
        save_path (str): The path to save the processed labeled image.
    Returns:
        None
    """

    filename = filename.split("/")[-1]
    orig, label = load_data(filename, orig_path, label_path)
    if label is not None:
        mask = create_mask(label)
        masked_orig = create_label(orig, mask)
        save_label(masked_orig, save_path, filename)


if __name__ == "__main__":
    orig_path = "/Path/to/orig_data"
    label_path = "/Path/to/label_data"
    save_path = "/Path/to/save_data"

    data_list = get_data_list(orig_path, save_path)

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(main, filename, orig_path, label_path, save_path)
                for filename in data_list
            ]

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                future.result()
