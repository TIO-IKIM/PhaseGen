# -*- coding: utf-8 -*-
"""Module containing building parts for `spectral_layer`, such as ComplexDropout or Spectral Pooling."""

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import math
from utils import utilities as u


class _ComplexBatchNorm(nn.Module):
    """
    Adapted from https://github.com/wavefrontshaping/complexPyTorch/tree/master.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    """Implements a batch normalization layer for complex inputs.
    It extends the functionality of _ComplexBatchNorm.
    Adapted from https://github.com/wavefrontshaping/complexPyTorch/tree/master.

    Args:
        track_running_stats (bool, optional): If True, running mean and covariance
                                               will be kept during training and
                                               used for normalization during inference.
        momentum (float, optional): Value used for exponential moving average.
        affine (bool, optional): If True, the layer will have learnable parameters,
                                 weight and bias.

    Shape:
        input: (N, C, H, W), where C is the number of channels and (H, W) are the
                 dimensions of the input feature map.
        output: (N, C, H, W), normalized complex feature map of the same shape.
    """

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean = input.mean([0, 2, 3])
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1.0 / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (
            Rrr[None, :, None, None] * input.real
            + Rri[None, :, None, None] * input.imag
        ).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * input.imag
            + Rri[None, :, None, None] * input.real
        ).type(
            torch.complex64
        )

        if self.affine:
            input = (
                self.weight[None, :, 0, None, None] * input.real
                + self.weight[None, :, 2, None, None] * input.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * input.real
                + self.weight[None, :, 1, None, None] * input.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )

        return input.cfloat()


class _complex_naive_batchnorm(nn.Module):
    """
    Implements a complex batch normalization layer for PyTorch. This layer normalizes complex-valued inputs
    by computing mean and covariance over the batch dimension.

    Args:
        num_features: Number of features in the input.
        dim (int): Dimension of the input data. 2 for 2D slices, 3 for 3D volumes. Defaults to 2.
        eps (optional): A small float added to the variance to avoid division by zero. Default to 1e-5.
        momentum (optional): The value used for the running_mean and running_covar computation. Defaults to 0.1.
        affine (optional): A boolean value that indicates whether to learn scale and shift parameters.
            Defaults to True.
        track_running_stats (optional): A boolean value that indicates whether to keep track of the running mean
            and covariance of the inputs. Defaults to True.

    Attributes:
        weight: A PyTorch Parameter that holds the scale parameters for the layer. Shape: (num_features, 3).
        bias: A PyTorch Parameter that holds the shift parameters for the layer. Shape: (num_features, 2).
        running_mean: A PyTorch Tensor that holds the running mean of the inputs. Shape: (num_features,).
        running_covar: A PyTorch Tensor that holds the running covariance of the inputs. Shape: (num_features, 3).
        num_batches_tracked: A PyTorch Tensor that holds the number of batches that were used to compute the
            running_mean and running_covar. Shape: (1,).
    """

    def __init__(
        self,
        num_features,
        dim: int = 2,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_complex_naive_batchnorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            if dim == 2:
                self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
                self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            elif dim == 3:
                self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
                self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer(
                "running_var", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.zero_()
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)


class ComplexNaiveBatchnorm(_complex_naive_batchnorm):
    """ComplexNaiveBatchNorm class inherits the _complex_naive_batchnorm class and implements forward pass.

    This implementation calculates mean of real and imaginary part of the input and variance. If running in training mode and
    track_running_stats is set to True, the mean and variance will be used to update the running mean and running variance.
    Then the input is normalized by subtracting mean and dividing it by the square root of variance plus epsilon. Finally,
    the normalized input is multiplied by the weight gamma and added with the bias beta to get the output.

    Attributes:
        training (bool): Flag to indicate if in training or evaluation mode
        track_running_stats (bool): Flag to indicate if to track running statistics or use the running statistics
        num_batches_tracked (int): Running count of number of batches processed
        momentum (float): The value used for updating running mean and variance
        eps (float): A small value added to the variance to prevent division by zero
        running_mean (torch.tensor): The running mean of the real and imaginary parts of the input
        running_var (torch.tensor): The running variance of the real and imaginary parts of the input
        gamma (torch.tensor): The weight to be multiplied to the normalized input
        beta (torch.tensor): The bias to be added to the multiplied input
    """

    def forward(self, input):
        assert input.ndim in [4, 5], f"Input must have dimension, but has {input.ndim}."

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            if input.ndim == 4:
                # 2D slices (B, C, H, W)
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3])
            if input.ndim == 5:
                # 3D volume (B, C, H, W, D)
                mean = input.mean([0, 2, 3, 4])
                var = input.var([0, 2, 3, 4])
        else:
            mean = self.running_mean
            var = self.running_var

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1.0 - self.momentum
                ) * self.running_var + self.momentum * var

        if input.ndim == 4:
            input = input - (mean / torch.sqrt(var + self.eps))[None, :, None, None]
        elif input.ndim == 5:
            input = (
                input - (mean / torch.sqrt(var + self.eps))[None, :, None, None, None]
            )

        input = self.gamma * input + self.beta

        return input


class SpectralPool(nn.Module):
    def __init__(self, kernel_size: int):
        """
        Initializes the SpectralPool module.

        SpectralPool is a module for spatial pooling in convolutional neural networks. It operates
        on input tensors and reduces their spatial dimensions while preserving important features.

        Args:
            kernel_size (int): Size of the pooling kernel. The input will be divided into grids of
                this size for pooling.

        Note:
            This module currently supports 2D input data (B, C, H, W) and 3D volume data is planned
            for future implementations.
        """
        super(SpectralPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the SpectralPool module.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled output tensor.
            torch.Tensor: Saved cut-off tensor.

        Note:
            - For 2D input data (B, C, H, W), this function performs spatial pooling using the
              provided kernel size.
            - The saved cut-off tensor represents the difference between the input and the pooled
              output.
        """
        if input.ndim == 4:
            # 2D slices (B, C, H, W)
            cut_off = int(math.ceil(input.size(3) // self.kernel_size))

            # Use torchvision's CenterCrop for pooling
            pooled_input = T.CenterCrop(size=(cut_off, cut_off))(input)

            # Pad pooled_input with zeros to match the size of input
            padded_output = u.padding(pooled_input, input)

            # Calculate saved_cut_off
            saved_cut_off = input - padded_output

        # TO-DO:
        # Implement new pooling for Volumes

        # elif input.ndim == 5:
        #     # 3d volume (B, C, H, W, D)
        #     pooled_shape = (input.size(2) // 2, input.size(3) // 2, input.size(4) // 2)
        #     pooled_input = torch.empty(pooled_shape, dtype=torch.cfloat)

        #     cut_off1 = math.ceil(input.size(2) / self.kernel_size)
        #     cut_off2 = math.ceil(input.size(3) / self.kernel_size)
        #     cut_off3 = math.ceil(input.size(4) / self.kernel_size)
        #     s1 = (input.size(2) - cut_off1) // 2
        #     s2 = (input.size(3) - cut_off2) // 2
        #     s3 = (input.size(4) - cut_off3) // 2

        #     pooled_input = input[..., s1:-s1, s2:-s2, s3:-s3]

        return pooled_input, saved_cut_off


class ComplexDropout(nn.Module):
    """
    A custom implementation of dropout for complex-valued tensors.

    Attributes:
        p (float): The dropout rate, a probability value between 0 and 1.
    """

    def __init__(self, p: float):
        """The constructor of ComplexDropout.

        Args:
            p (float): The dropout rate, a probability value between 0 and 1.
        """
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, input: Tensor):
        """
        If the input tensor is real-valued, dropout is performed using `nn.functional.dropout`.
        If the input tensor is complex-valued, a custom dropout is performed by creating a dropout
        mask for the real part of the tensor and applying it element-wise to the input.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after dropout.
        """
        if self.training:
            if input.is_complex():
                mask = nn.functional.dropout(torch.ones_like(input.real), self.p)
                return input * mask
            else:
                return nn.functional.dropout(input, self.p)
        else:
            return input


def complex_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs complex matrix multiplication between two tensors.

    This function takes two tensors, `a` and `b`, and performs complex matrix multiplication
    between them. The tensors are expected to have a shape of (batch_size, channels, height, width)
    or (batch_size, channels, length). The function reshapes `b` to match the shape of `a` and then
    performs matrix multiplication using Einstein summation (`torch.einsum`) with the appropriate
    contraction indices.

    Args:
        a (torch.Tensor): The first tensor for complex matrix multiplication.
        b (torch.Tensor): The second tensor for complex matrix multiplication.

    Returns:
        torch.Tensor: The result of complex matrix multiplication.
    """

    b = b.permute(1, 0, 2, 3)
    c = torch.einsum("aikl, bjkl -> ajkl", [a, b])

    return c
