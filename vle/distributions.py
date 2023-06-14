from vle.modules.functional import epsilon
from vle.utils import instantiate_from_config
from vle.modules.loss import setup_lpips_loss

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


def folded_normal_pdf(x, mu=0.0, sigma=1.0):
    sig_squared = sigma**2
    left_half = 1 / np.sqrt(2 * np.pi * sig_squared)
    left_half = left_half * np.exp(-1 * (x - mu) ** 2 / (2 * sig_squared))

    right_half = 1 / np.sqrt(2 * np.pi * sig_squared)
    right_half = right_half * np.exp(-1 * (x + mu) ** 2 / (2 * sig_squared))

    return left_half + right_half


def lr_lambda_fn(epoch):
    return 0.65**epoch


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        memory_network=None,
        max_tokens=10,
        update_token_dist_step=5000,
        learning_rate=1e-3,
        use_p_loss=True,
        p_mult=0.1,
        kld_mult=0.1,
    ):
        super().__init__()
        self.enc = instantiate_from_config(encoder)
        self.dec = instantiate_from_config(decoder)
        self.memory_network = None
        if memory_network is not None:
            self.memory_network = instantiate_from_config(memory_network)

        self.kld_mult = kld_mult
        self.p_mult = p_mult
        self.use_p_loss = use_p_loss
        self.lr = learning_rate

        self.lpips_loss = setup_lpips_loss() if use_p_loss else None

        self.max_tokens = max_tokens
        self.tokens_range = np.arange(1, max_tokens + 1)
        # Distribution to follow for selecting `n_tokens`
        self.token_choice_dist = folded_normal_pdf
        # Update distribution for token sampling after `update_token_dist_step`
        self.update_token_dist_step = update_token_dist_step

        self.downsample_factor = 2 ** (len(encoder["params"]["channel_mults"]) - 1)
        self.latent_channels = encoder["params"]["latent_channels"]

        self.mu_predictor = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.var_predictor = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def load_weights(self, ckpt_path, verbose=False):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict["state_dict"])
        if verbose:
            print("Weights sucessfully loaded!")

    def training_step(self, batch, batch_idx):
        # bs, ch, height, width = inputs.shape
        x = batch
        n_tokens = self.sample_n_tokens()

        kld_loss = 0.0
        int_rec_loss = 0.0
        prev_i_rec = None
        rec = torch.zeros_like(x)
        hidden_states = self.initialize_hidden_states(x)
        for _ in range(n_tokens):
            err = x - rec
            mean = torch.mean(err, dim=(-1, -2), keepdims=True)
            std = torch.std(err, dim=(-1, -2), keepdims=True) + epsilon
            delta = (err - mean) / std

            mu, log_var, hidden_states = self.encode(delta, hidden_states)
            z = self.reparameterize(mu, log_var)
            i_rec = self.decode(z) * std + mean

            # Compute intermediate losses
            mu, log_var = mu.flatten(start_dim=1), log_var.flatten(start_dim=1)
            kld_loss += torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            ) / torch.numel(rec[0])
            int_rec_loss += F.mse_loss(x, i_rec)

            rec = rec + i_rec
            prev_i_rec = i_rec

        if hasattr(self.enc, "reset_hidden_state"):
            self.enc.reset_hidden_state()

        # Compute final losses
        p_loss = 0.0
        if self.lpips_loss is not None:
            p_loss = self.lpips_loss(x, rec).mean() * self.p_mult
        loss = F.mse_loss(x, rec)
        kld_loss *= self.kld_mult

        # Average intermediate losses
        int_rec_loss /= n_tokens
        kld_loss /= n_tokens

        self.log(
            "n_tokens",
            torch.tensor(n_tokens).float(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("mse_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("kld_loss", kld_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "int_rec_loss", int_rec_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        if self.lpips_loss is not None:
            self.log("lpips_loss", p_loss, on_step=True, on_epoch=False, prog_bar=True)

        ep = min(self.global_step / self.update_token_dist_step, 0.5)
        return loss * ep + int_rec_loss * (1 - ep) + kld_loss + p_loss

    def sample_n_tokens(self):
        n_tokens_p = (
            self.token_choice_dist(
                self.tokens_range - 1,
                min(
                    int(self.global_step / self.update_token_dist_step), self.max_tokens
                ),
            )
            + epsilon
        )
        n_tokens_p = n_tokens_p / np.sum(n_tokens_p)
        n_tokens = np.random.choice(self.tokens_range, p=n_tokens_p)
        return n_tokens

    def initialize_hidden_states(self, x):
        if self.memory_network is None:
            return None

        ds_h = x.size(-2) // self.downsample_factor
        ds_w = x.size(-1) // self.downsample_factor

        shape = (x.size(0),) + (self.latent_channels,) + (ds_h, ds_w)
        hidden_states = self.memory_network.init_hidden(shape, self.device)
        return hidden_states

    def encode(self, x, hidden_states=None):
        result = self.enc(x)
        if hidden_states is not None:
            result, hidden_states = self.memory_network(result, hidden_states)
        mu = self.mu_predictor(result)
        log_var = self.var_predictor(result)
        return mu, log_var, hidden_states

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, shape, n_tokens):
        b, _, h, w = shape
        ds_h = h // self.downsample_factor
        ds_w = w // self.downsample_factor
        img_output = torch.zeros(shape).to(self.device)
        for _ in range(n_tokens):
            z = torch.randn(b, self.latent_channels, ds_h, ds_w)
            img_output = img_output + self.decode(z.to(self.device))
        return img_output

    def _forward(self, x, hidden_states, ret_z=False):
        mu, log_var, hidden_states = self.encode(x, hidden_states)
        z = self.reparameterize(mu, log_var)
        if ret_z:
            return self.decode(z), hidden_states, (mu, log_var, z)
        return self.decode(z), hidden_states

    def forward(self, x, n_tokens, ret_tokens=False, ret_z=False):
        tokens = [] if ret_tokens else None
        z_tokens = [] if ret_z else None
        rec = torch.zeros_like(x)
        hidden_states = self.initialize_hidden_states(x)
        for _ in range(n_tokens):
            err = x - rec
            mean = torch.mean(err, dim=(-1, -2), keepdims=True)
            std = torch.std(err, dim=(-1, -2), keepdims=True) + epsilon
            delta = (err - mean) / std

            if not ret_z:
                i_rec, hidden_states = self._forward(delta, hidden_states, ret_z=ret_z)
            else:
                i_rec, hidden_states, z = self._forward(
                    delta, hidden_states, ret_z=ret_z
                )
                z_tokens.append(z)

            i_rec = i_rec * std + mean
            rec = rec + i_rec

            if ret_tokens:
                tokens.append(i_rec)

        if hasattr(self.enc, "reset_hidden_state"):
            self.enc.reset_hidden_state()

        additional_ret_values = {}
        if tokens is not None:
            additional_ret_values["tokens"] = tokens
        if z_tokens is not None:
            additional_ret_values["z_tokens"] = z_tokens

        if len(additional_ret_values) > 0:
            return rec, additional_ret_values
        return rec

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiplicativeLR(
                    optimizer,
                    lr_lambda=lr_lambda_fn,
                ),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def log_img(self, batch, batch_idx, sample_latent=False, **kwargs):
        reconstructions = self(batch, self.max_tokens)
        reconstructions = reconstructions.moveaxis(1, -1)[:batch_idx]
        reconstructions = torch.hstack([r for r in reconstructions])
        log = {"reconstructions": reconstructions}
        if sample_latent:
            n_tokens = self.sample_n_tokens()
            samples = self.sample(batch.shape, n_tokens)
            samples = samples.moveaxis(1, -1)[:batch_idx]
            samples = torch.hstack([s for s in samples])
            log["samples"] = samples
        return log
