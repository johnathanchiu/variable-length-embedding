import math
import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 32, 128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)
        self.hack = nn.Linear(3200, 128)

    def forward(self, x):
        for block in self.enc_blocks:
            x = self.pool(block(x))
        return self.hack(torch.flatten(x, 1, -1))


class Decoder(nn.Module):
    def __init__(self, token_dim, image_embed_dim, chs=(128, 64, 32, 16, 4)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 3, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)]
        )

        self.final_conv = nn.Conv2d(chs[-1], 3, 4)

        self.token_dim = token_dim
        self.image_embed_dim = image_embed_dim

        self.reshape_size = int(math.sqrt(image_embed_dim // self.chs[0]))
        self.token_dec = nn.Linear(token_dim, image_embed_dim)

    def forward(self, x):
        x = torch.reshape(
            self.token_dec(x), (-1, self.chs[0], self.reshape_size, self.reshape_size)
        )
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = self.dec_blocks[i](x)

        return self.final_conv(x)


class TokenRNN(nn.Module):
    def __init__(self, token_dim, image_embed_dim):
        super().__init__()
        self.num_stacked_layers = 2
        self.token_dim = token_dim
        self.lstm = nn.LSTM(
            token_dim + image_embed_dim,
            token_dim,
            num_layers=self.num_stacked_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.x_initializer = nn.Linear(image_embed_dim, token_dim)
        self.token_out = nn.Linear(token_dim, token_dim)

    def forward(self, image_embed, x, hid_state):
        # x is prev token, hid_state is (h,c) hidden, cell state from previous cell
        x = torch.cat((x, image_embed), dim=1).unsqueeze(1)
        _, (hn, cn) = self.lstm(x, hid_state)
        out = self.token_out(hn[0])
        return out, (hn, cn)

    def initialize_inputs(self, image_embed):
        device = image_embed.get_device()
        bs = image_embed.shape[0]
        x = self.x_initializer(image_embed)
        h0 = torch.zeros(self.num_stacked_layers, self.token_dim).to(device)
        c0 = torch.zeros(self.num_stacked_layers, self.token_dim).to(device)
        h = h0.unsqueeze(1).repeat(1, bs, 1)
        c = c0.unsqueeze(1).repeat(1, bs, 1)
        return x, (h, c)
