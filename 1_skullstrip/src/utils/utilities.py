# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import logging
import matplotlib.pyplot as plt

plt.set_loglevel("info")
logging.getLogger("PIL").setLevel(logging.WARNING)
from prettytable import PrettyTable
import torch
import torchvision
from torch import Tensor
import numpy as np
import utils.fourier as FFT
from PIL import Image
import os
from typing import Iterable, Tuple, Union
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import functools
import time
from pathlib import Path
import models.layer.losses as losses
import random
import torch.nn.functional as F


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


def check_complex_accuracy(
    loader,
    model,
    loss_fn: torch.nn.Module,
    device: torch.cuda.device,
    transform=None,
    binary_label: bool = False,
    imagespace: bool = False,
) -> dict[str, float]:
    """Calculate various metrics for validation over the whole dataset presented by `loader`:
    - total loss: `total_loss`
    - absolute loss: `total_loss_abs`
    - angular loss: `total_loss_ang`
    - Dice score: `dsc`
    - Hausdorff distance: `hd`
    - Normalized mean squared error: `nmse`
    - Peak signal-to-noise ratio: `psnr`

    These scores are saved in the dictionary `metrics`.

    Args:
        loader (torch.utils.data.dataloader): The dataloader containing the slices or volumes to calculate the metrics on
        model (torch.model): The pytorch model to be validated.
        loss_fn (torch.Module): The loss to be applied.
        device (torch.cuda.device): The device to calculate the metrics on.
        transform (str): Preprocessign applied to the data. Here used for the backtransformation. Defaults to None.
        binary_label (bool): Whether the labels are already binary. If false, labels will be transformed into binary labels.
            Defaults to false.
        imagespace (bool): Whether the labels are in imagespace or fourier transformed. Defaults to False

    Returns:
        dict[str, float]: The dictionary explained above.
    """
    metrics = {
        "total_loss": 0,
        "total_loss_abs": 0,
        "total_loss_ang": 0,
        "dsc": 0,
        "hd": 0,
        "nmse": 0,
        "psnr": 0,
    }

    model.eval()
    nmse_loss = losses.NMSELoss()
    psnr_loss = losses.PSNR()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(x)
            loss = loss_fn(preds, y)
            loss_abs = loss_fn(preds.abs(), y.abs())
            loss_ang = loss_fn(preds.angle(), y.angle())
            metrics["total_loss"] += loss.item()
            metrics["total_loss_abs"] += loss_abs.item()
            metrics["total_loss_ang"] += loss_ang.item()

            if imagespace:
                preds_ifft = preds
                y_ifft = y
            else:
                preds_ifft = FFT.ifft(preds)
                y_ifft = FFT.ifft(y)

            dsc, hd = check_dice(preds_ifft, y_ifft, binary=binary_label)
            metrics["dsc"] += dsc
            metrics["hd"] += hd
            if preds.ndim == 4:
                metrics["nmse"] += nmse_loss(preds_ifft, y_ifft)
                metrics["psnr"] += psnr_loss(preds_ifft, y_ifft)
            elif preds.ndim == 5:
                for slice in range(preds_ifft.size(-1)):
                    metrics["nmse"] += nmse_loss(
                        preds_ifft[..., slice], y_ifft[..., slice]
                    )
                    metrics["psnr"] += psnr_loss(
                        preds_ifft[..., slice], y_ifft[..., slice]
                    )
                metrics["nmse"] /= slice
                metrics["psnr"] /= slice

    for i in metrics:
        metrics[i] /= len(loader)

    model.train()

    return metrics


def complex_save_predictions_as_imgs(
    loader,
    model,
    epoch: int,
    folder: str,
    device: torch.device,
    num: int = 4,
    kdiff: bool = False,
    iter: int = None,
    res: bool = False,
    res_dice: bool = False,
    shift: bool = False,
    binary_label: bool = False,
    imagespace: bool = False,
) -> None:
    """Visualize predictions for one iteration of a given dataset.

    This function performs prediction visualization for one iteration of a given dataset. If the folder specified does not
    exist, it will be created. The input to the model is transformed according to the given transformation argument and
    predictions are made. Then, plots are saved of the original and predicted values, in both k-space and image space.
    Additional outputs, such as k-space differences plots and Dice score plots, may also be saved if specified.

    Args:
        model (nn.Module): The model to be used for predictions.
        folder (str): The path of the folder where output images will be saved.
        device (str): The device to perform computations on (e.g. "cpu" or "cuda").
        transformation (str): The type of data transformation to be applied to the input.
        shift (bool): Whether or not to apply an FFT shift to the input and prediction data.
        res (bool): Whether or not to calculate residuals (i.e. input minus prediction).
        kdiff (bool): Whether or not to calculate and save k-space differences.
        res_dice (bool): Whether or not to calculate and save Dice scores for residual images.
        num (int): The number of images to visualize in each iteration.
        iter (int or None): The iteration number to visualize. If None, a random iteration will be selected.
        epoch (int): The current training epoch.
        loader (DataLoader): The PyTorch data loader for the input data.
        binary_label (bool): Whether the labels are already binary. If false, labels will be transformed into binary labels.
            Defaults to false.
        imagespace (bool): Whether the labels are in imagespace or fourier transformed. Defaults to False

    Returns:
        None.
    """
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    model.eval()
    if iter is not None:
        m = iter
    else:
        try:
            # Create random int in range of dataset as idx for batch to be visually evaluated
            m = np.random.randint(low=0, high=len(loader) - 1)
        except:
            m = 0
    # This needs some rewriting. Right now enumerating through whole dataset each time until idx == m
    for idx, (x, y) in enumerate(loader):
        if idx == m:
            if res == True:
                orig_strips = x[:num] - y[:num]
            else:
                orig_strips = y[:num]

            x = x[:num].to(device)

            with torch.no_grad():
                preds = model(x)

            orig_strips_ifft = FFT.ifft(orig_strips)

            if shift == True:
                preds = torch.fft.ifftshift(preds)
                x = torch.fft.ifftshift(x)
                y = torch.fft.ifftshift(y)
            if res == True:
                pred_strip = x - preds
                preds_ifft = FFT.ifft(preds)
            else:
                pred_strip = preds
            pred_strip = pred_strip.to("cpu")
            if imagespace == False:
                pred_strip_ifft = FFT.ifft(pred_strip)
            x_ifft = FFT.ifft(x)
            if preds.ndim == 5:
                slice = random.randint(5, 165)
                pred_strip = pred_strip[..., slice]
                x = x[..., slice]
                orig_strips = orig_strips[..., slice]
                orig_strips_ifft = orig_strips_ifft[..., slice]
                x_ifft = x_ifft[..., slice]
                pred_strip_ifft = pred_strip_ifft[..., slice]
                if res == True:
                    preds_ifft = preds_ifft[..., slice]

            # Choose random channel to be plotted for evaluation
            c = random.randint(0, preds.size(1) - 1)

            if epoch % 1 == 0 or epoch == 0:
                if imagespace == False:
                    outputs = [
                        (x_ifft.abs()[:, 0, None, ...], f"{folder}/orig_{epoch}.png"),
                        (
                            torch.log(orig_strips.abs()[:, c, None, ...]),
                            f"{folder}/orig_seg_kspace_{epoch}.png",
                        ),
                        (
                            torch.log(pred_strip.abs()[:, c, None, ...]),
                            f"{folder}/pred_seg_kspace_{epoch}.png",
                        ),
                        (
                            orig_strips_ifft.abs()[:, c, None, ...],
                            f"{folder}/{epoch}.png",
                        ),
                        (
                            pred_strip_ifft.abs()[:, c, None, ...],
                            f"{folder}/pred_seg_{epoch}.png",
                        ),
                    ]
                elif imagespace == True:
                    outputs = [
                        (x_ifft.abs()[:, 0, None, ...], f"{folder}/orig_{epoch}.png"),
                        (
                            orig_strips[:, c, None, ...].float(),
                            f"{folder}/orig_seg_{epoch}.png",
                        ),
                        (
                            pred_strip.abs()[:, c, None, ...],
                            f"{folder}/pred_seg_{epoch}.png",
                        ),
                    ]
                for output in outputs:
                    torchvision.utils.save_image(output[0], output[1], normalize=True)
            if res == True:
                torchvision.utils.save_image(
                    preds_ifft.abs()[:, c, ...],
                    f"{folder}/pred_res_{epoch}.png",
                    normalize=True,
                )
            if kdiff == True:
                for i in range(num):
                    kspace_difference(pred_strip[i], orig_strips[i], epoch, i, folder)

            break

    model.train()


def dice(input: float, target: float, smooth: int = 1) -> float:
    """Calculates the DICE score between the input and target.

    Args:
        input (float): The input or prediction data.
        target (float): The target or ground truth data.
        smooth(int): Smoothing factor.

    Returns:
        float: The calculated DICE score.
    """
    intersection = np.sum(input * target)
    return (2.0 * intersection + smooth) / (np.sum(input) + np.sum(target) + smooth)


def hausdorff(input: float, target: float) -> float:
    """Calculate Directed Hausdorff Distance.

    Args:
        input (float): Input (Prediction)
        target (float): Target (Ground Truth)

    Returns:
        float: The calculated DHD score.
    """
    HD = directed_hausdorff(input, target)[0]
    return HD


def check_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: int = 1,
    seg_perc: float = 2.0,
    binary: bool = False,
) -> Tuple[float, float]:
    """Calculate the DICE coefficient (DSC) and Hausdorff distance (HD) between a predicted slice or volume and the corresponding label.

    If the predictions is a 3D volume, the DSC and HD are calculate for each slice and then averaged.
    The number of non-zero pixels in the target segmentation is counted and compared to the `threshold`.
    Is the number of non-zero pixels lower than the threshold, the corresponding slice is not considered for calculation,
    as there is not enough information in the image.

    For DICE and HD calculation, the input tensors are transformed into binary masks. The prediction binary masks is calculated
    with the threshold `seg_perc` times the mean value of the prediction to filter out noisy background due to the
    nature of the fourier domain. Holes in both segmentation masks are then filled with the `ndimage.binary_fill_holes)` function.
    The final binary masks are given to the functions `hausdorff()` and `dice()` to calculate the scores.

    If prediction and target are not on the CPU, copy them to the CPU.

    Args:
        pred (torch.Tensor): The complex valued predicted segmentation in image space.
        target (torch.Tensor): The complex valued target segmentation in image space
        threshold (int, optional): Non-zero pixel threshold for filtering the top and bottom segmentations. Defaults to 1.
        seg_perc (float, optional): Percentage to multiply with the mean value of the prediction to filter noise. Defaults to 2.0.
        binary (bool): Whether the labels are already binary. If false, labels will be transformed into binary labels.
            Defaults to false.

    Returns:
        Tuple[float, float]: The calculated Dice score and Hausdorff distance respectively.
    """

    dice_score, HD = 0, 0

    if pred.ndim not in [4, 5]:
        raise ValueError(f"Dimensions of input need to be 4 or 5, but is {pred.ndim}!")

    if pred.device.type == target.device.type != "cpu":
        pred = pred.to(device="cpu", non_blocking=True)
        target = target.to(device="cpu", non_blocking=True)

    if pred.ndim == 4:
        samples_calculated = 0
        # Iterate trough samples in batch
        for i in range(target.size(0)):
            target_ = target[i, 0, ...]
            if binary == False:
                binary_target = torch.Tensor(
                    (target_.abs() > target_.abs().mean()).float()
                ).numpy()
                # binary_target = ndimage.binary_fill_holes(binary_target.numpy()).astype(
                #     float
                # )
            else:
                binary_target = target_.abs().numpy().astype(float)
            if (np.count_nonzero(binary_target) < threshold) == False:
                # ignore scans containing no brain for DSC calculation
                binary_pred = torch.Tensor(
                    (
                        pred[i, 0, ...].abs() > pred[i, 0, ...].abs().mean() * seg_perc
                    ).float()
                )
                binary_pred = ndimage.binary_fill_holes(binary_pred.numpy()).astype(
                    float
                )
                _HD = hausdorff(binary_pred, binary_target)
                _dice_score = dice(binary_pred, binary_target)

                if not np.isnan(_dice_score):
                    HD += _HD
                    dice_score += _dice_score
                    samples_calculated += 1

        if HD != 0:
            HD /= samples_calculated
        if dice_score != 0:
            dice_score /= samples_calculated

    elif pred.ndim == 5:
        slices_calculated = 0
        for slice in range(pred.size(-1)):
            # Iterate trough samples in batch
            for i in range(target.size(0)):
                target_ = target[i, 0, ..., slice]
                if np.count_nonzero(target_) > threshold:
                    if binary == False:
                        binary_target = torch.Tensor(
                            (target_).abs() > target_.abs().mean().float()
                        )
                        binary_target = ndimage.binary_fill_holes(
                            binary_target.numpy()
                        ).astype(float)
                    else:
                        binary_target = target_.abs().numpy().astype(float)
                    binary_pred = torch.Tensor(
                        (
                            pred[i, 0, ..., slice].abs()
                            > pred[i, 0, ..., slice].abs().mean() * seg_perc
                        ).float()
                    )
                    binary_pred = ndimage.binary_fill_holes(binary_pred.numpy()).astype(
                        float
                    )

                    HD += hausdorff(binary_pred, binary_target)
                    dice_score += dice(binary_pred, binary_target)
                    slices_calculated += 1

        HD /= slices_calculated
        dice_score /= slices_calculated

    return dice_score, HD


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """
    https://github.com/fkodom/fft-conv-pytorch/blob/master/fft_conv_pytorch/fft_conv.py
    Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.
    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple
    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""

    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def diff(folder: str, epoch: int) -> torch.tensor:
    """Calculate difference between ground truth and output.

    Args:
        folder (str): Folder of Dice images

    Returns:
        torch.tensor: Difference between original scan and strip
    """
    convertTensor = torchvision.transforms.ToTensor()
    orig = convertTensor(Image.open(f"{folder}/{epoch}.png"))
    brain_strip = convertTensor(Image.open(f"{folder}/pred_seg_{epoch}.png"))

    diff = orig[0, :, :] - brain_strip[0, :, :]

    return diff


def accuracy(binary_pred, binary_target):
    """
    Calculate confusion matrix, accuracy, sensitivity & specificity of binary skull strips.

    Args:
        binary_pred [numpy_array], binary_target [tensor]: binary arrays of skull strips (ground truth & prediction)

    Returns:
        Accuracy, sensitivity & specificity
    """
    fp = len(np.where(binary_target - binary_pred == -1)[0])
    fn = len(np.where(binary_target - binary_pred == 1)[0])
    tp = len(np.where(binary_target + binary_pred == 2)[0])
    tn = len(np.where(binary_target + binary_pred == 0)[0])
    total = fp + fn + tp + tn

    accuracy = (tp + tn) / total
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return accuracy, specificity, sensitivity


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


def kspace_difference(
    pred: Tensor, orig: Tensor, epoch: int, i: int, folder: str
) -> None:
    """Plot original an predicted absolute kspace, as well as the difference map with colorbar.

    Args:
        pred (Tensor): predicted kspace data
        orig (Tensor): original kspace data
        epoch (int): current epoch
        i (int): current iteration (in case of testing)
        folder (str, optional): Folder to save image in.
    """
    kspace_difference = torch.log(orig) - torch.log(pred)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5), dpi=500)
    cmap = "Greys"
    orig = ndimage.rotate(torch.log(orig.abs())[0, :, :], 90)
    pred = ndimage.rotate(torch.log(pred.abs())[0, :, :], 90)
    ax1.imshow(orig, cmap=cmap)
    ax2.imshow(pred, cmap=cmap)
    kspace_difference = ndimage.rotate(kspace_difference.abs()[0, :, :], 90)
    im = ax3.imshow(kspace_difference, cmap=cmap)
    fig.subplots_adjust(right=0.85)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax1.set_title("Ground Truth k-space", fontsize=16)
    ax2.set_title("Predicted k-space", fontsize=16)
    ax3.set_title("Difference in k-space", fontsize=16)
    fig.colorbar(
        im, ax=(ax1, ax2, ax3), pad=0.02, aspect=10, shrink=0.9, cmap=cmap, label="log"
    )
    fig.savefig(f"{folder}/kspace_difference_{epoch}_{i}.png")
    plt.close()


def phase_difference(
    pred: Tensor, orig: Tensor, epoch: int, i: int, folder: str
) -> None:
    """Plot original an predicted phase in image space, as well as the difference map with colorbar.

    Args:
        pred (Tensor): predicted phase data
        orig (Tensor): "original" phase data
        epoch (int): current epoch
        i (int): current iteration (in case of testing)
        folder (str, optional): Folder to save image in.
    """
    difference = orig - pred
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    cmap = "Greys"
    ax1.imshow(orig.angle()[0, :, :], cmap=cmap)
    ax2.imshow(pred.angle()[0, :, :], cmap=cmap)
    im = ax3.imshow(difference.angle()[0, :, :], cmap=cmap)
    # divider = make_axes_locatable((ax3))
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.subplots_adjust(right=0.85)
    # cax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, ax=(ax1, ax2, ax3), pad=0.02, aspect=10, shrink=0.9, cmap=cmap)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax1.set_title("Original Input phase", fontsize=16)
    ax2.set_title("Predicted phase", fontsize=16)
    ax3.set_title("Difference in phase", fontsize=16)
    fig.savefig(f"{folder}/phase_difference_{epoch}_{i}.png")
    plt.close()


@torch.jit.script
def to_complex(real, imag):
    """Concatenate real and imaginary part to complex tensor

    Args:
        real (torch.tensor): Real part of complex number.
        imag (torch.tensor): Imaginary part of complex number
    Return:
        Complex tensor.
    """
    complex = real + 1j * imag

    return complex.cfloat()


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
        print(f"Finished {func.__name__!r} in {run_time:.3f} secs")
        return value

    return wrapper_timer


def tensor_to_image(input: torch.Tensor):
    """Convert a PyTorch tensor to a PIL image.

    Args:
    input (torch.Tensor): Input tensor with shape (Batch, Channel, Height, Width), (Channel, Height, Width) or (Height, Width).

    Returns:
    list[Image]: A list of PIL images. One for each sample in the input tensor.
    """

    images = []

    if input.dim() == 4:
        for sample in input:
            if sample.is_complex():
                sample = sample.abs()
            sample = sample.squeeze(0)
            sample_normed = (
                (sample - sample.min()) / (sample.max() - sample.min()) * 255
            )
            image = Image.fromarray(sample_normed)
            images.append(image)
    elif input.dim() == 3:
        if input.is_complex():
            input = input.abs()
        input = input.squeeze(0)
        input_normed = (input - input.min()) / (input.max() - input.min()) * 255
        image = Image.fromarray(input_normed)
        images.append(image)
    else:
        if input.is_complex():
            input = input.abs()
        input_normed = (input - input.min()) / (input.max() - input.min()) * 255
        image = Image.fromarray(input_normed)
        images.append(image)

    return images


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

    pad_h = original.size(2) - pooled_input.size(2)
    pad_w = original.size(3) - pooled_input.size(3)
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    padded = F.pad(pooled_input, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))

    return padded
