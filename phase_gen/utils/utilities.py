# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torchvision
import numpy as np
import os
import functools
import time
import random
import torch.nn.functional as F


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            self.logger.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).",
            )
        self.val_score_min = val_score
        self.save = True


def complex_diffusion_accuracy(
    loader,
    model,
    device: torch.cuda.device,
) -> dict[str, float]:
    """
    Evaluates the accuracy of a complex diffusion model on a given dataset.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict[str, float]: A dictionary containing the evaluation metrics, specifically the validation loss.
    """
    metrics = {
        "val_loss": 0,
    }

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)

            loss = model(x)
            metrics["val_loss"] += loss.item()

    for i in metrics:
        metrics[i] /= len(loader)

    model.train()

    return metrics


def get_diffusion_sample(
    loader,
    model,
    epoch: int,
    folder: str,
    device: torch.device,
    num: int = 4,
    iter: int = None,
) -> None:
    """
    Generates and saves diffusion model samples for visual evaluation.

    Args:
        loader (DataLoader): DataLoader providing the dataset.
        model (torch.nn.Module): The diffusion model to generate samples from.
        epoch (int): The current epoch number, used for naming the output files.
        folder (str): The directory where the output images will be saved.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        num (int, optional): The number of samples to generate. Defaults to 4.
        iter (int, optional): Specific iteration index to use for sampling. If None, a random index is chosen. Defaults to None.

    Returns:
        None
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
            x = x[:num].to(device)

            with torch.no_grad():
                preds = model.sample(x).to("cpu")

            # Choose random channel to be plotted for evaluation
            c = random.randint(0, preds.size(1) - 1)

            if epoch % 1 == 0 or epoch == 0:
                outputs = [
                    (x.abs()[:, 0, None, ...], f"{folder}/input_abs_{epoch}.png"),
                    (x.angle()[:, 0, None, ...], f"{folder}/target_phase_{epoch}.png"),
                    (
                        preds.angle()[:, c, None, ...],
                        f"{folder}/pred_phase_{epoch}.png",
                    ),
                    (preds.abs()[:, c, None, ...], f"{folder}/pred_abs_{epoch}.png"),
                ]
                for output in outputs:
                    torchvision.utils.save_image(output[0], output[1], normalize=True)

            break

    model.train()


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
