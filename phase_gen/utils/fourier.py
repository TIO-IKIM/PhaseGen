# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch


@torch.jit.script
def ifft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """
    Performs an inverse Fast Fourier Transform (iFFT) on the input tensor along specified dimensions.
    Args:
        scan (torch.Tensor): The input tensor to be transformed. Must have 2, 3, 4, or 5 dimensions.
        dim (tuple[int, int], optional): The dimensions along which to perform the iFFT. Defaults to (-2, -1).
    Returns:
        torch.Tensor: The transformed tensor after applying iFFT.
    Raises:
        ValueError: If the input tensor does not have 2, 3, 4, or 5 dimensions.
    """

    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )
    scan = torch.fft.ifftshift(scan, dim=dim)
    scan = torch.fft.ifft2(scan, dim=dim)
    scan = torch.fft.fftshift(scan, dim=dim)

    return scan


@torch.jit.script
def fft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """
    Performs an Fast Fourier Transform (FFT) on the input tensor along specified dimensions.
    Args:
        scan (torch.Tensor): The input tensor to be transformed. Must have 2, 3, 4, or 5 dimensions.
        dim (tuple[int, int], optional): The dimensions along which to perform the FFT. Defaults to (-2, -1).
    Returns:
        torch.Tensor: The transformed tensor after applying FFT.
    Raises:
        ValueError: If the input tensor does not have 2, 3, 4, or 5 dimensions.
    """

    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )

    scan = torch.fft.ifftshift(scan, dim=dim)
    scan = torch.fft.fft2(scan, dim=dim)
    scan = torch.fft.fftshift(scan, dim=dim)

    return scan
