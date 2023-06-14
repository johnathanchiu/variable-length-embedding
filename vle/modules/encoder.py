from vle.modules.layers import ResidualBlock, Downsample
from vle.utils import instantiate_from_config

import torch.nn.functional as F
import torch.nn as nn
import torch


class Encoder(nn.Module):
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

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))

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

        self.encoder_conv_out = torch.nn.Conv2d(
            now_channels,
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

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
        return x


class MemoryEncoder(nn.Module):
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

        self.activation = activation
        self.initial_pad = initial_pad

        self.memory_network = None
        if memory_network is not None:
            assert base_channels == memory_network["params"]["input_channels"]
            self.memory_network = instantiate_from_config(memory_network)

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()

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

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))

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

        self.encoder_conv_out = torch.nn.Conv2d(
            now_channels,
            latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def reset_hidden_state(self):
        if self.memory_network is not None:
            self.memory_network.reset_hidden_state()

    def set_n_tokens(self, n_tokens):
        if self.memory_network is not None:
            self.memory_network.set_n_tokens(n_tokens)

    def forward(self, x):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        x = self.init_conv(x)

        if self.memory_network is not None:
            x = self.memory_network(x)

        for layer in self.downs:
            x = layer(x)

        for layer in self.mid:
            x = layer(x)

        x = self.encoder_conv_out(x)
        return x
