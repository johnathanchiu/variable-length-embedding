from vle.modules.functional import postprocess, inverse_input_norm, input_norm, epsilon
from vle.modules.layers import ResidualBlock
from vle.utils import instantiate_from_config
from vle.modules.loss import (
    LPIPSLoss,
    mse_with_mask,
    masks_diff_loss,
    scaled_sigmoid,
    scaled_tanh,
)

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
        learning_rate=1e-3,
        use_p_loss=True,
        update_token_dist_step=5000,
        sigmoid_annealing_steps=100,
        sigmoid_scale=10.0,
        max_tokens=10,
        min_tokens=1,
        p_mult=0.1,
        mask_mult=0.3,
    ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder)
        self.decoder = instantiate_from_config(decoder)

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

        self.lpips_loss = LPIPSLoss() if use_p_loss else None

        self.p_mult = p_mult
        self.use_p_loss = use_p_loss
        self.mask_mult = mask_mult
        self.lr = learning_rate
        self.sigmoid_scale = sigmoid_scale

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokens_range = np.arange(min_tokens, max_tokens + 1)
        # Distribution to follow for selecting `n_tokens`
        self.token_choice_dist = folded_normal_pdf
        # Update distribution for token sampling after `update_token_dist_step`
        self.update_token_dist_step = update_token_dist_step
        self.sigmoid_annealing_steps = sigmoid_annealing_steps

    def load_weights(self, ckpt_path, verbose=False):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict["state_dict"])
        if verbose:
            print("Weights sucessfully loaded!")

    def training_step(self, batch, batch_idx):
        # bs, ch, height, width = inputs.shape
        n_tokens = self.sample_n_tokens()
        _, _, loss_dict = self(batch, n_tokens, compute_losses=True)
                
        self.log(
            "n_tokens",
            torch.tensor(n_tokens).float(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            self.log(loss_name, loss_value, on_step=True, on_epoch=False, prog_bar=True)
            total_loss += loss_value

        return total_loss

    def sample_n_tokens(self):
        n_tokens_p = (
            self.token_choice_dist(
                self.tokens_range - self.min_tokens,
                min(
                    int(self.global_step / self.update_token_dist_step), self.max_tokens
                ),
            )
            + epsilon
        )
        n_tokens_p = n_tokens_p / np.sum(n_tokens_p)
        n_tokens = np.random.choice(self.tokens_range, p=n_tokens_p)
        return n_tokens

    def encode(self, x):
        result =  self.encoder(x)
        mu = self.mu_predictor(result)
        log_var = self.var_predictor(result)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        x,
        n_tokens,
        compute_losses=False,
        log=False,
    ):
        if log:
            intermediate_tokens = []
            intermediate_masks = []

        if compute_losses:
            aux_mask_loss = 0.0
            aux_rec_loss = 0.0
            aux_kld_loss = 0.0

        # Set number of tokens for submodules
        self.set_n_tokens(n_tokens)

        rec = torch.zeros_like(x)
        mask_shape = (x.size(0), 1, x.size(2), x.size(3))
        mask = torch.zeros(mask_shape, device=self.device)
        nil_mask = torch.zeros_like(mask, requires_grad=False)
        for i in range(n_tokens):
            # residuals = x - rec
            model_input = torch.cat([x, mask], dim=1).contiguous()
            mu, log_var = self.encode(model_input)
            z = self.reparameterize(mu, log_var)
            model_output = self.decode(z)
            # Decode and split apart reconstruction from mask
            i_rec, i_mask = torch.split(model_output, (3, 1), dim=1)
            i_mask = torch.sigmoid(i_mask)

            if compute_losses:
                aux_rec_loss += mse_with_mask(x, i_rec, i_mask)
                empty_mask_loss = masks_diff_loss(i_mask, nil_mask)
                mask_separation_loss = masks_diff_loss(i_mask, mask)
                aux_mask_loss += empty_mask_loss + mask_separation_loss
                mu, log_var = mu.flatten(start_dim=1), log_var.flatten(start_dim=1)
                aux_kld_loss += torch.mean(
                    -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
                ) / torch.numel(rec[0])

            mask = torch.tanh(mask + i_mask)
            rec = rec + i_rec

            if log:
                intermediate_masks.append(i_mask.clamp(0.0, 1.0))
                intermediate_tokens.append(
                    inverse_input_norm(i_rec.clamp(-1.0, 1.0))
                )
        if log:
            intermediate_masks.append(i_mask.clamp(0.0, 1.0))

        # Reset hidden states for submodules
        self.reset_hidden_state()

        if compute_losses:
            # p_loss = 0.0
            # if self.lpips_loss is not None:
            #     p_loss = self.lpips_loss(rec, x)
            rec_loss = F.mse_loss(rec, x)

        # Return values
        if log:
            intermediates = {
                "model_output": intermediate_tokens,
                "mask_output": intermediate_masks,
            }
            rec = inverse_input_norm(rec.clamp(-1.0, 1.0))
            return rec, mask, intermediates

        if compute_losses:
            loss_dict = {
                "aux_rec_loss": aux_rec_loss / n_tokens,
                "aux_mask_loss": aux_mask_loss / n_tokens,
                "kld_loss": aux_kld_loss / n_tokens,
                # "mask_loss": mask_loss,
                "rec_loss": rec_loss,
                # "lpips_loss": p_loss,
            }
            return rec, mask, loss_dict

        return rec, mask
    
    def set_n_tokens(self, n_tokens):
        if hasattr(self.encoder, "set_n_tokens"):
            self.encoder.set_n_tokens(n_tokens)

    def reset_hidden_state(self):
        if hasattr(self.encoder, "reset_hidden_state"):
            self.encoder.reset_hidden_state()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.MultiplicativeLR(
            #         optimizer,
            #         lr_lambda=lr_lambda_fn,
            #     ),
            #     "monitor": "train_loss",
            #     "frequency": 1,
            # },
        }

    @torch.no_grad()
    def log_img(self, batch, batch_idx, sample_intermediates=False, **kwargs):
        ret_values = self(
            batch,
            self.max_tokens,
            log=sample_intermediates,
        )
        if sample_intermediates:
            reconstructions, _, intermediates = ret_values
        else:
            reconstructions, _ = ret_values

        rec = postprocess(reconstructions)
        log = {"reconstructions": [[r] for r in rec]}
        for key, value in intermediates.items():
            reformatted_tokens = []
            for token in value:
                token = postprocess(token)
                reformatted_tokens.append(token)
            reformatted_tokens = [
                [i for i in t] for t in reformatted_tokens
            ]
            log[key] = reformatted_tokens
        return log
