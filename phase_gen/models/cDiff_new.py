import torch
import torch.nn.functional as F
from models.cResUNet import ResUNet
import math


class PhaseDiffusionUNet(ResUNet):

    def __init__(
        self,
        config,
        features,
        device,
        activation,
        padding=None,
        dilation: int = 1,
        in_channels: int = 2,
        out_channels: int = 1,
        timesteps=1000,
    ):
        """
        Initializes the PhaseDiffusionUNet.

        Args:
            config (dict): Configuration dictionary.
            features (int): Number of features.
            device (torch.device): Device to run the model on.
            activation (str): Activation function.
            padding (str, optional): Padding type. Defaults to None.
            dilation (int, optional): Dilation rate. Defaults to 1.
            in_channels (int, optional): Number of input channels. Defaults to 2.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            timesteps (int, optional): Number of timesteps for the diffusion process. Defaults to 1000.
        """
        super(PhaseDiffusionUNet, self).__init__(
            config,
            features,
            device,
            activation,
            padding,
            dilation,
            in_channels,
            out_channels,
        )

        self.timesteps = timesteps
        self.in_channels = in_channels

        betas = self._cosine_variance_schedule(timesteps, 0.008)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = ResUNet(
            config,
            features,
            device,
            activation,
            padding,
            dilation,
            in_channels,
            out_channels,
        )

    def loss_fn(
        self,
        pred_noise: torch.Tensor,
        actual_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the L1 loss between predicted noise and actual noise.

        Args:
            pred_noise (torch.Tensor): Predicted noise tensor.
            actual_noise (torch.Tensor): Actual noise tensor.

        Returns:
            torch.Tensor: Computed L1 loss.
        """

        return F.l1_loss(pred_noise, actual_noise)

    def _cosine_variance_schedule(
        self, timesteps: int, epsilon: float = 0.008
    ) -> torch.Tensor:
        """
        Generates a cosine variance schedule.

        Args:
            timesteps (int): Number of timesteps.
            s (float): Scaling factor.

        Returns:
            torch.Tensor: Beta values for the variance schedule.
        """

        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0, 0.999)

        return betas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PhaseDiffusionUNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after normalization and passing through the network.
        """

        x = (x - x.mean(dim=[1, 2, 3], keepdim=True)) / x.std(
            dim=[1, 2, 3], keepdim=True
        )

        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        noise = 1 * torch.exp(1j * torch.randn_like(x).angle())
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(torch.cat([x_t, x.abs()], dim=1), t)

        loss = self.loss_fn(pred_noise, noise, x_t, x)

        return loss

    def _forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the forward diffusion process to the input tensor `x_0` at time step `t` with added noise.

        Args:
            x_0 (torch.Tensor): The original input tensor.
            t (torch.Tensor): The time step tensor.
            noise (torch.Tensor): The noise tensor to be added to the input.

        Returns:
            torch.Tensor: The noised tensor after applying the forward diffusion process.
        """

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(
            x_0.shape[0], 1, 1, 1
        )
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(x_0.shape[0], 1, 1, 1)

        noised_x = (sqrt_alpha_cumprod_t * x_0) + (
            sqrt_one_minus_alphas_cumprod * noise
        )

        return noised_x

    def sample(self, input: torch.Tensor) -> torch.Tensor:
        """
        Samples a new tensor based on the input tensor using a reverse diffusion process.

        Args:
            input (torch.Tensor): The input tensor to be sampled from. It is expected to be a 4D tensor.

        Returns:
            torch.Tensor: The sampled tensor after applying the reverse diffusion process.

        The function performs the following steps:
        1. Converts the input tensor to its absolute values and casts it to float.
        2. Whitens the input tensor by subtracting its mean and dividing by its standard deviation along the specified dimensions.
        3. Initializes a complex tensor `x_t` by multiplying the whitened input tensor with a complex exponential of random angles.
        4. Iteratively applies the reverse diffusion process for a specified number of timesteps.
        5. In each iteration, generates a noise tensor with random angles and updates `x_t` using the `_reverse_diffusion` method.
        6. Finally, returns the input tensor multiplied by a complex exponential of the angles of `x_t`.
        """
        input = input.abs().float()
        input_whitened = (input - input.mean(dim=[1, 2, 3], keepdim=True)) / input.std(
            dim=[1, 2, 3], keepdim=True
        )

        x_t = input_whitened * torch.exp(1j * torch.randn_like(input_whitened).angle())

        for i in reversed(range(0, self.timesteps, 1)):
            noise = 1 * torch.exp(1j * torch.randn_like(x_t).angle())
            t = torch.tensor([i for _ in range(input.shape[0])]).to(self.device)

            x_t = self._reverse_diffusion(x_t, t, noise, input_whitened)

        return input * torch.exp(1j * x_t.angle())

    def _reverse_diffusion(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the reverse diffusion process to generate a sample from the model.

        Args:
            x_t (torch.Tensor): The current state tensor at time step t.
            t (torch.Tensor): The time step tensor.
            noise (torch.Tensor): The noise tensor to be added during the reverse diffusion process.
            magnitude (torch.Tensor): The magnitude tensor to be concatenated with x_t.

        Returns:
            torch.Tensor: The generated sample tensor after applying the reverse diffusion process.
        """

        pred = self.model(torch.cat([x_t, magnitude], dim=1), t)

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(x_t.shape[0], 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1 - alpha_t_cumprod_prev) / (1 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise
