from vle.modules.layers import ResidualBlock
from vle.utils import instantiate_from_config

import torch.nn as nn
import torch


class SalinecyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        out_channels=1,
        memory_network=None,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
    ):
        super().__init__()
        self.memory_network = None
        if memory_network is not None:
            assert base_channels == memory_network["params"]["input_channels"]
            self.memory_network = instantiate_from_config(memory_network)
        
        layers = []
        now_channels = base_channels
        for _, mult in enumerate(channel_mults):
            mid_channels = base_channels * mult

            for _ in range(num_res_blocks):
                layers.append(
                    ResidualBlock(
                        now_channels,
                        mid_channels,
                        dropout=0.1,
                        num_groups=1,
                    ),
                )
                now_channels = mid_channels

        self.saliency = nn.Sequential(*layers)
        self.conv_in = nn.Conv2d(
            in_channels, 
            base_channels, 
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_out = torch.nn.Conv2d(
            now_channels,
            out_channels,
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
        x = self.conv_in(x)
        
        if self.memory_network is not None:
            x = self.memory_network(x)

        x = self.saliency(x)
        return self.conv_out(x)
    

class SimpleSaliencyModel(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels=1,
        memory_network=None,
    ):
        super().__init__()
        self.memory_network = None
        if memory_network is not None:
            assert mid_channels == memory_network["params"]["input_channels"]
            self.memory_network = instantiate_from_config(memory_network)

        self.input_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            dropout=0.1,
            num_groups=1,
        )
        self.output_block = ResidualBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            dropout=0.1,
            num_groups=1,
        )

    
    def reset_hidden_state(self):
        if self.memory_network is not None:
            self.memory_network.reset_hidden_state()

    def set_n_tokens(self, n_tokens):
        if self.memory_network is not None:
            self.memory_network.set_n_tokens(n_tokens)

    def forward(self, x):
        x = self.input_block(x)
        
        if self.memory_network is not None:
            x = self.memory_network(x)

        return self.output_block(x)