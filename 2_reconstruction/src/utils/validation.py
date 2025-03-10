from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from typing import Optional
import json


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def nrmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Root Mean Squared Error (NRMSE)"""
    return np.sqrt(nmse(gt, pred))


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:

    maxval = gt.max() if maxval is None else maxval

    return structural_similarity(gt, pred, data_range=maxval)


class ReconstructionEvaluation:
    """
    A class to evaluate reconstruction quality using various metrics.

    Attributes:
        metrics : dict
            A dictionary to store the computed metrics: Mean Squared Error (mse),
            Normalized Mean Squared Error (nmse), Peak Signal-to-Noise Ratio (psnr),
            and Structural Similarity Index (ssim).

    Methods
        __compute_metrics(gt: np.ndarray, pred: np.ndarray)
            Computes the mse, nmse, psnr, and ssim metrics for the given ground truth
            and prediction arrays and updates the metrics dictionary.
        __call__(gt: np.ndarray, pred: np.ndarray)
            Evaluates the reconstruction by computing the metrics for the given ground
            truth and prediction arrays. Supports both 3D and 4D arrays.
        __getitem__(key)
            Returns the value of the specified metric from the metrics dictionary.
    """

    def __init__(self, normalize: bool = False, verbose: bool = False):
        self.metrics = {
            "mse": [],
            "nrmse": [],
            "psnr": [],
            "ssim": [],
        }
        self.normalize = normalize
        self.verbose = verbose

    @staticmethod
    def _normalize(gt: np.ndarray, pred: np.ndarray):

        gt = (gt - gt.min()) / (gt.max() - gt.min())
        pred = (pred - pred.min()) / (pred.max() - pred.min())

        return gt, pred

    def _compute_metrics(
        self, gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
    ):

        maxval = gt.max() if maxval is None else maxval
        gt = gt.squeeze()
        pred = pred.squeeze()
        self.metrics["ssim"].append(ssim(gt, pred, maxval))
        self.metrics["psnr"].append(psnr(gt, pred))
        self.metrics["mse"].append(mse(gt, pred))
        self.metrics["nrmse"].append(nrmse(gt, pred).item())

    def _save_metrics(self, filename="metrics.json"):
        """Save all computed metrics to a JSON file."""
        metrics_to_save = {k: np.array(v).tolist() for k, v in self.metrics.items()}
        with open(filename, "w") as f:
            json.dump(metrics_to_save, f, indent=4)

    def __call__(self, pred: np.ndarray, gt: np.ndarray):

        assert (
            gt.shape == pred.shape
        ), "Ground truth and prediction must have same shape."
        assert gt.ndim in [
            3,
            4,
        ], "Ground truth and prediction must have 3 or 4 dimensions."

        if gt.ndim == 4:
            for i in range(gt.shape[0]):
                if self.normalize:
                    gt[i], pred[i] = self._normalize(gt[i], pred[i])
                maxval = gt.max()
                self._compute_metrics(gt[i], pred[i], maxval=maxval)
        else:
            if self.normalize:
                gt, pred = self._normalize(gt, pred)
            self._compute_metrics(gt, pred)

    def __getitem__(self, key):

        if self.verbose:
            self._save_metrics()

        if key == "all":
            return {k: np.array(v).mean() for k, v in self.metrics.items()}
        else:
            return np.array(self.metrics[key]).mean()
