# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import models.layer.spectral_blocks as layer
import models.layer.spectral_layer as parts


class ResUNet(nn.Module):
    """
    ResUNet is a residual U-Net model for image processing tasks.

    Attributes:
        downs (nn.ModuleList): List of downsampling blocks.
        bottleneck (nn.ModuleList): List of bottleneck blocks.
        ups (nn.ModuleList): List of upsampling blocks.
        pooling_size (int): Size of the pooling kernel.
        pool (nn.Module): Pooling layer.
        padding (int): Padding size.
        kernel_size (int): Size of the convolutional kernel.
        device (torch.device): Device to run the model on.
        dropout (float): Dropout rate.
        dilation (int): Dilation rate.
        activation (nn.Module): Activation function.
        res_length (int): Length of the residual blocks.
        final_layer (nn.Conv2d): Final convolutional layer.
    """

    def __init__(
        self,
        config: dict,
        features: list,
        device: torch.device | list[torch.device],
        activation: nn.Module,
        padding: int = None,
        dilation: int = 1,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        """
        Initializes the ResUNet model.

        Args:
            config (dict): Configuration dictionary.
            features (list): List of feature sizes for each block.
            device (torch.device | list[torch.device]): Device(s) to run the model on.
            activation (nn.Module): Activation function.
            padding (int, optional): Padding size. Defaults to None.
            dilation (int, optional): Dilation rate. Defaults to 1.
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
        """
        super(ResUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pooling_size = config["pooling_size"]
        self.pool = parts.SpectralPool(kernel_size=self.pooling_size)
        self.padding = padding
        self.kernel_size = config["kernel_size"]

        self.device = device
        self.dropout = config["dropout"]
        self.dilation = dilation
        self.activation = activation
        conv = nn.Conv2d
        self.res_length = config["length"]

        # Create downsampling blocks
        for feature in features:
            self.downs.append(
                layer.ResidualBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    dropout=self.dropout,
                    kernel_size=self.kernel_size,
                    device=self.device,
                    padding=self.padding,
                    stride=1,
                    dilation=self.dilation,
                    activation=self.activation,
                    resample=True,
                    embedding=True,
                )
            )
            in_channels = feature
            for i in range(self.res_length):
                self.downs.append(
                    layer.ResidualBlock(
                        in_channels=feature,
                        out_channels=feature,
                        dropout=self.dropout,
                        kernel_size=self.kernel_size,
                        device=self.device,
                        padding=self.padding,
                        stride=1,
                        dilation=self.dilation,
                        activation=self.activation,
                        resample=False,
                    )
                )

        # Create bottleneck block
        bottleneck_in_channels = features[-1]
        bottleneck_out_channels = features[-1] * 2
        self.bottleneck.append(
            layer.ResidualBlock(
                in_channels=bottleneck_in_channels,
                out_channels=bottleneck_out_channels,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
                device=self.device,
                padding=self.padding,
                stride=1,
                dilation=self.dilation,
                activation=self.activation,
                resample=True,
            )
        )

        # Create upsampling block
        for feature in reversed(features):
            self.ups.append(
                layer.Upsampling(
                    in_channels=feature * 2,
                    out_channels=feature,
                    device=self.device,
                    scale_factor=self.pooling_size,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                )
            )
            self.ups.append(
                layer.ResidualBlock(
                    in_channels=feature * 2,
                    out_channels=feature,
                    dropout=0,
                    kernel_size=self.kernel_size,
                    device=self.device,
                    padding=self.padding,
                    stride=1,
                    dilation=self.dilation,
                    activation=self.activation,
                    resample=True,
                    embedding=True,
                )
            )

        # Create final layer
        self.final_layer = conv(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            device=self.device,
            dtype=torch.complex64,
        )

    def pos_encoding(self, t, channels):
        """
        Generates positional encoding for the given time steps and channels.

        Args:
            t (torch.Tensor): Time steps tensor.
            channels (int): Number of channels.

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        t = t.unsqueeze(-1).type(torch.float)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t=None):
        """
        Forward pass of the ResUNet model.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor, optional): Time steps tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """

        skip_connections = []

        t = self.pos_encoding(t, 256)

        # Downsample blocks
        for idx, down in enumerate(self.downs):
            x = down(x, t)

            # Save intermediate outputs after downsampling
            if idx % (self.res_length + 1) == 0:
                skip_connections.append(x)
                x, _ = self.pool(x)

        # Bottleneck block
        for bottleneck in self.bottleneck:
            x = bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsample blocks
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Ensure output size matches skip_connection output size
            if x.shape != skip_connection.shape:
                x_real = nn.Upsample(size=skip_connection.shape[2:])(x.real)
                x_imag = nn.Upsample(size=skip_connection.shape[2:])(x.imag)
                x = x_real + 1j * x_imag

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip, t)

        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)

        # Final layer
        x = self.final_layer(x)

        if x.ndim == 5:
            x = x.permute(0, 1, 3, 4, 2)

        return x
