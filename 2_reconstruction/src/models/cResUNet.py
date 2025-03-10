# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import torch.nn as nn
import models.layer.spectral_blocks as layer
import models.layer.spectral_layer as parts
from utils.fourier import ifft, fft


class ResUNet(nn.Module):
    """Implementation of a complex valued Residual U-Net model.

    Args:
        config (dict): Configuration dictionary.
        features (list): List of feature channels for each layer.
        device (torch.device): Device for computation.
        activation (nn.Module): Activation function for the model.
        padding (int, optional): Padding size. Defaults to None.
        dilation (int, optional): Dilation rate for convolutional layers. Defaults to 1.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        logger: Logger for debugging. Defaults to None.
        viz (bool, optional): Flag for visualization. Defaults to False.

    Attributes:
        downs (nn.ModuleList): List of downsampling layers.
        bottleneck (nn.ModuleList): List of bottleneck layers.
        ups (nn.ModuleList): List of upsampling layers.
        pool: Spectral pooling layer.
        final_layer: Final convolutional layer.

    Methods:
        forward(x): Forward pass of the ResUNet model.

    Note:
        The model is designed for complex-valued inputs.
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
        image_domain: bool = False,
    ) -> None:
        super(ResUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pooling_size = config["pooling_size"]
        self.pool = parts.SpectralPool(kernel_size=self.pooling_size)
        self.padding = padding
        self.kernel_size = config["kernel_size"]
        self.consistency = True
        self.image_domain = image_domain

        # Set default padding if not provided
        if padding == None:
            self.padding = int((self.kernel_size - 1) / 2)

        self.device = device
        self.dropout = config["dropout"]
        self.dilation = dilation
        self.activation = activation
        Conv = nn.Conv2d
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
                    image_domain=self.image_domain,
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
                )
            )

        # Create final layer
        self.final_layer = nn.Sequential(
            Conv(
                in_channels=features[0],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                device=self.device,
                dtype=torch.cfloat
            ),
        )
        
    def get_mask(self, x_in):
        mask = (x_in.abs() > 0).float()
        return mask
        
    def data_consistency(self, x_pred, x_in):
        """Data consistency layer for the model.

        Args:
            x_pred (torch.Tensor): Predicted output.
            x_in (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Consistent output.
        """
        if self.image_domain:
            x_pred = fft(x_pred)
        mask = self.get_mask(x_in)
        
        # Scale x_in and x_pred
        x_in = x_in * mask
        x_pred = x_pred * (1 - mask)
        
        scale_factor = (x_pred.abs().mean() + 1e-6) / (x_in.abs().mean() + 1e-6)
        x_in = x_in * scale_factor

        x_out = x_in + x_pred
        if self.image_domain:
            x_out = ifft(x_out)
            
        return x_out

    def forward(self, x, kspace_undersampled):

        skip_connections = []
        cut_offs = []
        if self.consistency:
            x_in_list = []
            x_in = kspace_undersampled
            x_in_list.append(x_in)

        # Downsample blocks
        for idx, down in enumerate(self.downs):
            x = down(x)
            
            # Save intermediate outputs after downsampling
            if idx % (self.res_length + 1) == 0:
                skip_connections.append(x)

                if self.image_domain:
                    x = fft(x)
                x, cut_off = self.pool(x)
                if self.image_domain:
                    x = ifft(x)
                if self.consistency:
                    x_in, _ = self.pool(x_in)
                    x_in_list.append(x_in) 
                    x = self.data_consistency(x, x_in)
                cut_offs.append(cut_off)

        # Bottleneck block
        for bottleneck in self.bottleneck:
            x = bottleneck(x)
        skip_connections = skip_connections[::-1]
        cut_offs = cut_offs[::-1]
        if self.consistency:
            x_in_list = x_in_list[::-1]

        # Upsample blocks
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x, cut_offs[idx // 2])
            skip_connection = skip_connections[idx // 2]

            # Ensure output size matches skip_connection output size
            if x.shape != skip_connection.shape:
                x_real = nn.Upsample(size=skip_connection.shape[2:])(x.real)
                x_imag = nn.Upsample(size=skip_connection.shape[2:])(x.imag)
                x = x_real + 1j * x_imag

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x = self.final_layer(x)

        return x
