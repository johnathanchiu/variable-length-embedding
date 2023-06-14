from vle.modules.layers import ResidualBlock, Upsample, get_norm
from vle.utils import instantiate_from_config

import torch.nn.functional as F
import torch.nn as nn
import torch


class Decoder(nn.Module):
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
        self.ups = nn.ModuleList()

        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                now_channels = out_channels

        self.activation = activation
        self.initial_pad = initial_pad

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
        x = self.decoder_conv_in(x)

        for layer in self.ups:
            x = layer(x)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            ip = self.initial_pad
            return x[:, :, ip:-ip, ip:-ip]
        return x
    

class MemoryDecoder(nn.Module):
    def __init__(
        self,
        img_channels,
        base_channels,
        latent_channels,
        memory_network=None,
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
        self.ups = nn.ModuleList()

        now_channels = base_channels * channel_mults[-1]

        self.activation = activation
        self.initial_pad = initial_pad

        self.decoder_conv_in = torch.nn.ConvTranspose2d(
            latent_channels,
            now_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.memory_network = None
        if memory_network is not None:
            memory_network["params"]["input_channels"] = now_channels
            self.memory_network = instantiate_from_config(memory_network)

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

    def reset_hidden_state(self):
        if self.memory_network is not None:
            self.memory_network.reset_hidden_state()

    def set_n_tokens(self, n_tokens):
        if self.memory_network is not None:
            self.memory_network.set_n_tokens(n_tokens)

    def forward(self, x):
        x = self.decoder_conv_in(x)

        if self.memory_network is not None:
            x = self.memory_network(x)  

        for layer in self.ups:
            x = layer(x)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            ip = self.initial_pad
            return x[:, :, ip:-ip, ip:-ip]
        return x
