# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import matplotlib.pyplot as plt

plt.set_loglevel("info")
from prettytable import PrettyTable
import torch
from utils.fourier import ifft
from typing import Tuple, Union
import functools
import time
from pathlib import Path
import torch.nn.functional as F
import logging


def save_checkpoint(state, save_folder: Union[str, Path], epoch: int) -> None:
    """Save checkpoint of network at given epoch.

    Args:
        state: State of the network, including optimizer, epoch, weights, etc.
        save_folder (str | Path): Path to save-folder for the checkpoint.
        epoch (int): Current epoch during which checkpoint is saved.

    Returns:
        None.
    """
    print("=> Saving checkpoint")
    torch.save(state, f"{save_folder}/checkpoint_{epoch}.pth.tar")


def load_checkpoint(checkpoint, model):
    """Load the checkpoint of a given model.

    Args:
        checkpoint: The checkpoint of the model which will be loaded.
        model (torch.nn.Module): The corresponding model structure of the checkpoint.

    Returns:
        None.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    return model


def plot_multicoil_example(
    kspace: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    epoch: int,
    save_path: Union[str, Path],
) -> None:
    """
    Plots and saves images of k-space input, target, and prediction for MRI reconstruction.
    Args:
        kspace (torch.Tensor): The k-space data tensor of shape (batch_size, num_coils, height, width).
        target (torch.Tensor): The ground truth image tensor of shape (batch_size, height, width).
        prediction (torch.Tensor): The predicted image tensor of shape (batch_size, height, width).
        epoch (int): The current epoch number, used for naming the saved files.
        save_path (Union[str, Path]): The directory path where the images will be saved.
    Returns:
        None
    """

    # plot first 4 coils of input in image domain for 4 examples
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    input_imge = ifft(kspace).abs().cpu().numpy()
    for j in range(4):
        for i in range(4):
            axs[j, i].imshow(input_imge[j, i, ...], cmap="gray")
            axs[j, i].axis("off")
            axs[j, i].set_title(f"Coil {i+1}")
    fig.savefig(f"{save_path}/input_epoch{epoch}.png")

    # plot target in image domain
    target = target.squeeze().cpu().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        axs = axs.flatten()
        axs[i].imshow(target[i], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Target Example {i+1}")
    fig.savefig(f"{save_path}/target_epoch{epoch}.png")

    # plot prediction in image domain
    prediction = prediction.squeeze().cpu().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        axs = axs.flatten()
        axs[i].imshow(prediction[i], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Prediction Example {i+1}")
    fig.savefig(f"{save_path}/prediction_epoch{epoch}.png")


def plot_singlecoil_example(
    input_image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    epoch: int,
    save_path: Union[str, Path],
    image_domain: bool,
) -> None:

    # plot 4 examples of input in image domain
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    input_image = input_image.squeeze()
    for i in range(4):
        axs = axs.flatten()
        axs[i].imshow(input_image[i], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Input Example {i+1}")
    fig.savefig(f"{save_path}/input_epoch{epoch}.png")
    plt.close()

    # plot target in image domain
    target = target.squeeze()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        axs = axs.flatten()
        axs[i].imshow(target[i], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Target Example {i+1}")
    fig.savefig(f"{save_path}/target_epoch{epoch}.png")
    plt.close()

    # plot prediction in image domain
    prediction = prediction.squeeze()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        axs = axs.flatten()
        axs[i].imshow(prediction[i], cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Prediction Example {i+1}")
    fig.savefig(f"{save_path}/prediction_epoch{epoch}.png")
    plt.close()


def count_parameters(model: torch.nn.Module) -> Tuple[PrettyTable, int]:
    """Counts the model parameters.

    Args:
        model (torch.nn.Module): a torch model

    Returns:
        int: number of model parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def timer(func):
    """Decorator that measures the runtime of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        wrapper_timer (function): The decorated function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__name__!r} in {run_time:.3f} secs")
        return value

    return wrapper_timer


def padding(pooled_input, original):
    """Pad a pooled input tensor to match the size of an original tensor.

    This function pads the 'pooled_input' tensor to match the spatial dimensions
    (height and width) of the 'original' tensor. It calculates the amount of padding
    required on each side and applies it symmetrically.

    Args:
        pooled_input (torch.Tensor): The pooled input tensor to be padded.
        original (torch.Tensor): The original tensor whose spatial dimensions
            the 'pooled_input' tensor should match.

    Returns:
        torch.Tensor: The padded 'pooled_input' tensor with the same spatial
        dimensions as the 'original' tensor.
    """

    pad_h = original.size(-2) - pooled_input.size(-2)
    pad_w = original.size(-1) - pooled_input.size(-1)
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    padded = F.pad(pooled_input, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))

    return padded


def normalize(gt, pred):

    gt = (gt - gt.min()) / (gt.max() - gt.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    return gt, pred
