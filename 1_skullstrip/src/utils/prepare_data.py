# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from pathlib import Path
from typing import Tuple
import fourier as fft
import os
import torch
from IKIMLogger import IKIMLogger
import nibabel as nib
import torch.nn.functional as F


def load_nifti(path: str) -> torch.Tensor:
    """Load nifti volumes and convert them into pytorch tensors.

    Args:
        path (str): Path to the nifti files.

    Returns:
        torch.Tensor: Return a pytorch tensor of the volume.
    """
    nifti_image = nib.load(path)
    nifti_image_data = nifti_image.get_fdata()
    tensor = torch.tensor(nifti_image_data)

    return tensor


def rescale_and_pad(
    data: torch.Tensor,
    shape: Tuple,
    rescaling: bool = False,
    label: torch.Tensor = None,
    phase: torch.Tensor = None,
    permute: Tuple = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rescale and pad data and corresponding label if provided according to the given `shape`.
    The Tensors are rescaled by first calculating the required scaling factors for height and width.
    The lower scaling factor is used such that the maximum value for height or width does not go over the desired shape.
    This might lead to a resized tensor which only has the correct depth and either width or height.
    The last dimension is then padded accordingly.

    Args:
        data (torch.Tensor): The input data to be resized and padded.
        shape (Tuple, optional): The desired shape for `data` and `label`.
        label (torch.Tensor, optional): The corresponding label or segmentation. Defaults to None.
        phase (torch.Tensor, optional): Phase data to be resized and padded. Defaults to None.
        permute (Tuple): Permutation of the tensor if dimensions need to be reordered. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The resized and padded `data` and `label`.
            If no label or phase is given, the last two returned values are `None`.
    """

    # Rescaling
    if permute:
        data = data.permute(permute)
        label = label.permute(permute)
    if rescaling:
        height_scale = shape[0] / int(data.size(0))
        width_scale = shape[1] / int(data.size(1))
        scale = min(height_scale, width_scale)
        h = round(int(data.size(0)) * scale)
        w = round(int(data.size(1)) * scale)
        up = torch.nn.Upsample(size=(h, w, shape[2]), mode="trilinear")
        data = up(data[None, None, ...])[0, 0, ...]
        if label is not None:
            label = up(label[None, None, ...])[0, 0, ...]
        if phase is not None:
            phase = up(phase[None, None, ...])[0, 0, ...]

    # Padding
    print(f"Shape before padding: {tuple(data.size())}.")
    h = (shape[0] - int(data.size(0))) / 2
    w = (shape[1] - int(data.size(1))) / 2
    h1, h2 = int(h), int(h)
    w1, w2 = int(w), int(w)
    if h % 1 != 0:
        h1 += 1
    if w % 1 != 0:
        w1 += 1
    data = F.pad(input=data, pad=(0, 0, w1, w2, h1, h2))

    if label is not None:
        label = F.pad(input=label, pad=(0, 0, w1, w2, h1, h2))
    if phase is not None:
        phase = F.pad(input=phase, pad=(0, 0, w1, w2, h1, h2))

    return data, label, phase


class CCPreprocessing:
    """
    Class for preprocessing CC-359 dataset, including resizing, masking, and saving 2D slices.

    Attributes:
        orig_path (Path): Path to the original images.
        label_path (Path): Path to the label images.
        orig_output_path (Path): Output path for processed original images.
        label_output_path (Path): Output path for processed label images.
        name_addition (str): Additional name for label images.
        resize_shape (Tuple): Desired shape for resizing.

    Methods:
        _get_patient(self, n: int, patient: str) -> None:
            Load original and label images for a given patient.

        _binary_to_image(self):
            Apply binary mask to the original image.

        _get_folder_len(folder: Path) -> int:
            Get the number of files in a folder.

        process_patients2d(self):
            Process patients' 2D slices and save the results.

        _save_processed_slice(self, patient: str, slice: int):
            Save processed slices.
    """

    def __init__(
        self,
        orig_path: Path,
        label_path: Path,
        orig_output_path: Path,
        label_output_path: Path,
        name_addition: str = None,
        resize_shape: Tuple = (256, 256, 200),
    ) -> None:
        self.orig_path = orig_path
        self.label_path = label_path
        self.orig_output_path = orig_output_path
        self.label_output_path = label_output_path
        self.resize_shape = resize_shape
        self.name_addition = name_addition

        logger.info(f"Desired output shape: {resize_shape}.")

    def _get_patient(self, n: int, patient: str) -> None:
        """
        Load original and label images for a given patient.

        Args:
            n (int): Patient index.
            patient (str): Patient name.

        Returns:
            None
        """
        logger.info(
            f"Patient {n+1} / {CCPreprocessing._get_folder_len(self.orig_path)}"
        )
        self.orig = load_nifti(f"{Path(self.orig_path, patient)}.nii.gz")

        self.label = load_nifti(
            f"{Path(self.label_path)}/{patient}_{self.name_addition}.nii.gz"
        )

        assert self.orig.size(2) == self.label.size(
            2
        ), f"Scan has {self.orig.size(2)} slices, while Strip has {self.label.size(2)} slices!"
        logger.info(f"Original shape: {tuple(self.orig.size())}.")

    def _binary_to_image(self):
        """
        Apply binary mask to the original image.

        Returns:
            None
        """
        self.strip = self.orig * self.strip

    def _get_folder_len(folder: Path) -> int:
        """
        Get the number of files in a folder.

        Args:
            folder (Path): Path to the folder.

        Returns:
            int: Number of files.
        """
        return len(os.listdir(folder))

    def process_patients2d(self) -> None:
        """
        Process patients' 2D slices and save the results.

        Returns:
            None
        """
        for n, patient in enumerate(os.scandir(self.orig_path)):
            try:
                patient = patient.name.split(".")[0]
                CCPreprocessing._get_patient(self, n, patient)

                self.orig, self.strip, phase = rescale_and_pad(
                    self.orig,
                    shape=(self.resize_shape + (self.orig.size(2),)),
                    label=self.label,
                )

                # Confidence mask to binary mask
                self.strip = torch.Tensor(self.strip > 0.3).float()

                CCPreprocessing._binary_to_image(self)

                assert tuple(self.orig.size()) == self.resize_shape + (
                    self.orig.size(2),
                ), (
                    f"Shape of original scan is not equal the desired shape of {tuple(self.resize_shape)}"
                    f", but {tuple(self.orig.size())} instead!"
                )
                assert (
                    self.orig.size() == self.strip.size()
                ), f"Shape of original scan [{self.orig.size()}] is not equal shape of skull strip [{self.strip.size()}]!"

                for slice in range(self.orig.size(2)):
                    self.orig_slice = self.orig[..., slice]
                    self.strip_slice = self.strip[..., slice]

                    self.orig_slice = fft.fft(self.orig_slice)
                    self.strip_slice = fft.fft(self.strip_slice)

                    CCPreprocessing._save_processed_slice(self, patient, slice)

            except Exception as e:
                logger.info(e)
                continue

    def _save_processed_slice(self, patient: str, slice: int):
        """
        Save processed slices.

        Args:
            patient (str): Patient name.
            slice (int): Slice index.

        Returns:
            None
        """
        torch.save(
            torch.rot90(self.orig_slice.detach().clone(), 1, [0, 1]),
            Path(f"{self.orig_output_path / patient}_slice_{slice}.pt"),
        )
        torch.save(
            torch.rot90(self.strip_slice.detach().clone(), 1, [0, 1]),
            Path(f"{self.label_output_path / patient}_slice_{slice}.pt"),
        )


if __name__ == "__main__":
    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir=Path.home() / "k-Strip" / "logs",
        comment="Data_preparation",
    )
    logger = ikim_logger.create_logger()

    orig_path = Path("path/to/original_data")
    label_path = Path("path/to/labels")
    orig_output_path = Path("path/to/save_transformed_orig")
    label_output_path = Path("path/to/save_transformed_label")
    name_addition = "staple"  # Name addition in case the label have special name appendix, e.g. "staple"

    resize_shape = (256, 256)
    processor = CCPreprocessing(
        orig_path,
        label_path,
        orig_output_path,
        label_output_path,
        name_addition,
        resize_shape=resize_shape,
    )

    processor.process_patients2d()
