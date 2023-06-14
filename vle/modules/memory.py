from vle.modules.layers import ConvLSTMCell

import torch.nn as nn
import torch


class ConvLSTM(nn.Module):
    def __init__(
        self,
        n_layers,
        input_channels,
        hidden_channels,
        use_token_embedder=False,
        max_tokens=16,
    ):
        super().__init__()

        self.hidden_state = None
        self.n_tokens = max_tokens
        self.use_token_embedder = use_token_embedder

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        if use_token_embedder:
            self.n_token_embedding = nn.Embedding(max_tokens, hidden_channels * 2)

        cell_list = []
        for i in range(n_layers):
            in_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    in_channels,
                    hidden_channels,
                )
            )
        self.cells = nn.ModuleList(cell_list)
        self.output_conv = nn.Conv2d(
            hidden_channels,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = self.init_hidden_state(x)
        o, h, c = x, *self.hidden_state
        for cell in self.cells:
            o, h, c = cell(o, h, c)
        o = self.output_conv(o)
        self.hidden_state = (h, c)
        return o

    def reset_hidden_state(self):
        self.hidden_state = None

    def set_n_tokens(self, n_tokens):
        self.n_tokens = n_tokens

    def init_hidden_state(self, x):
        shape, device = x.size(), x.get_device()
        shape = (shape[0],) + (self.hidden_channels,) + (shape[-2], shape[-1])

        if self.use_token_embedder:
            n_tokens = torch.tensor([self.n_tokens], device=device)
            z = self.n_token_embedding(n_tokens)[..., None, None]
            z_cell, z_hidden = z.split((self.hidden_channels, self.hidden_channels), dim=1)
            cell_state = z_cell.repeat((shape[0], 1, shape[-2], shape[-1]))
            hidden_state = z_hidden.repeat((shape[0], 1, shape[-2], shape[-1]))
        else:
            hidden_state = torch.zeros(shape, device=device)
            cell_state = torch.zeros(shape, device=device)

        return hidden_state, cell_state


if __name__ == "__main__":
    model = ConvLSTM(
        n_layers=1,
        input_channels=4,
        hidden_channels=16,
    ).cuda()
    x = torch.randn((1, 4, 64, 64)).cuda()
    hidden = model.init_hidden(x.shape, x.device)
    o, (h, c) = model(x, hidden)
    print(o.shape, h.shape, c.shape)
