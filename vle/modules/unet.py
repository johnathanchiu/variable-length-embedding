from vle.modules.layers import ResidualBlock, Upsample, Downsample, get_norm

import torch.nn.functional as F
import torch.nn as nn
import torch


# https://github.com/abarankab/DDPM/blob/main/ddpm/unet.py
class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
        self,
        img_channels,
        base_channels,
        latent_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        self.encoder_conv_out = torch.nn.Conv2d(
            now_channels,
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=True,
                ),
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=False,
                ),
            ]
        )

        self.ups = nn.ModuleList()

        self.decoder_conv_in = torch.nn.ConvTranspose2d(
            latent_channels,
            now_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        x = self.init_conv(x)

        for layer in self.downs:
            x = layer(x)

        for layer in self.mid:
            x = layer(x)

        x = self.encoder_conv_out(x)
        x = self.decoder_conv_in(x)

        for layer in self.ups:
            x = layer(x)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x


if __name__ == "__main__":
    from vle.modules.encoder import Encoder
    from vle.modules.decoder import Decoder
    kwargs = {
        "img_channels": 3,
        "base_channels": 32,
        "latent_channels": 4,
        "num_groups": 1,
        "channel_mults": [1, 2, 4, 4, 4, 4],
    }
    encoder = Encoder(**kwargs).cuda()
    decoder = Decoder(**kwargs).cuda()

    x = torch.randn((1, 3, 512, 512)).to("cuda")
    x_latent = encoder(x)
    print(x_latent.shape)
    y = decoder(x_latent)
    print(y.shape)

    # model = UNet(3, 32, 4).cuda()
    # x = torch.randn((1, 3, 512, 512)).to("cuda")
    # print(model(x).shape)
