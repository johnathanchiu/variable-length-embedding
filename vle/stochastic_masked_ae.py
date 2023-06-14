from vle.utils import instantiate_from_config
from vle.modules.loss import LPIPSLoss, mse_with_mask
from vle.modules.functional import (
    ClampLayer,
    RoundLayer,
    normalize,
    postprocess, 
    inverse_input_norm, 
    folded_normal_pdf,
    epsilon,
)

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


def lr_lambda_fn(epoch):
    return 0.65**epoch


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        learning_rate=1e-3,
        max_tokens=10,
        min_tokens=1,
        update_token_dist_step=5000,
        use_p_loss=False,
        p_mult=0.1,
        mask_mult=0.3,
    ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder)
        self.decoder = instantiate_from_config(decoder)
        self.rounder = RoundLayer()
        self.clamper = ClampLayer()

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

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokens_range = np.arange(min_tokens, max_tokens + 1)
        # Distribution to follow for selecting `n_tokens`
        self.token_choice_dist = folded_normal_pdf
        # Update distribution for token sampling after `update_token_dist_step`
        self.update_token_dist_step = update_token_dist_step
        

    def load_weights(self, ckpt_path, verbose=False):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict["state_dict"])
        if verbose:
            print("Weights sucessfully loaded!")

    def training_step(self, batch, batch_idx):
        # bs, ch, height, width = inputs.shape
        n_tokens = self.sample_n_tokens()
        *_, loss_dict = self(batch, n_tokens, compute_losses=True)
                
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
        z = self.encoder(x)
        mu = self.mu_predictor(z)
        log_var = self.var_predictor(z)
        return mu, log_var

    def decode(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_latent_shape(self, x):
        b, _, h, w = x.shape
        ds_h = h // self.downsample_factor
        ds_w = w // self.downsample_factor
        return b, self.latent_channels, ds_h, ds_w
    
    def sample(self, x, n_tokens):
        img = torch.zeros_like(x).to(self.device)
        latent_shape = self.compute_latent_shape(x)
        self.set_n_tokens(n_tokens)
        for _ in range(n_tokens):
            z = torch.randn(*latent_shape, device=self.device)
            img = img + self.decoder(z)
        self.reset_hidden_state()
        return img 

    def forward(self, x, n_tokens, compute_losses=False, log=False):
        if log:
            intermediate_tokens = []
            intermediate_masks = []

        if compute_losses:
            aux_rec_loss = 0.0
            aux_dist_loss = 0.0
            kld_loss = 0.0

        # Set number of tokens for submodules
        self.set_n_tokens(n_tokens)

        rec = torch.zeros_like(x)
        latent_shape = self.compute_latent_shape(x)
        latent_mu = torch.zeros(*latent_shape, device=self.device)
        for i in range(n_tokens):
            residuals, mean, std = normalize(x - rec)
            i_mu, i_log_var = self.encode(residuals)
            i_rec = self.decode(i_mu, i_log_var)
            i_rec = i_rec * std + mean

            if compute_losses:
                # aux_rec_loss += mse_with_mask(i_rec, x, i_mask)
                # Compute KLD loss
                i_mu_flat = i_mu.flatten(start_dim=1)
                i_log_var_flat = i_log_var.flatten(start_dim=1)
                kld_loss += torch.mean(
                    -0.5 * torch.sum(1 + i_log_var_flat - i_mu_flat**2 - i_log_var_flat.exp(), dim=1), dim=0
                ) / torch.numel(rec[0])
                # Independence term
                if i > 0:
                    aux_dist_loss += torch.mean(torch.sum(latent_mu * i_mu, dim=(1, 2, 3))) / torch.numel(rec[0])
                
            # Update masks, reconstructions
            rec = rec + i_rec
            latent_mu = latent_mu + i_mu

            if log:
                # intermediate_masks.append(i_mask.clamp(0.0, 1.0))
                intermediate_tokens.append(
                    inverse_input_norm(i_rec.clamp(-1.0, 1.0))
                )

        # Reset hidden states for submodules
        self.reset_hidden_state()

        if compute_losses:
            p_loss = 0.0
            if self.lpips_loss is not None:
                p_loss = self.lpips_loss(rec, x)
            rec_loss = F.mse_loss(rec, x)

        # Return values
        if log:
            intermediates = {
                "intermediate_reconstructions": intermediate_tokens,
                # "intermediate_masks": intermediate_masks,
            }
            generation = self.sample(rec, n_tokens)
            generation = inverse_input_norm(generation.clamp(-1.0, 1.0))
            rec = inverse_input_norm(rec.clamp(-1.0, 1.0))
            return rec, generation, intermediates

        if compute_losses:
            loss_dict = {
                # "aux_rec_loss": aux_rec_loss / n_tokens,
                "aux_distr_loss": aux_dist_loss / n_tokens,
                "kld_loss": kld_loss / n_tokens,
                "rec_loss": rec_loss,
                "lpips_loss": p_loss,
            }
            return rec, None, loss_dict

        return rec, None
    
    def set_n_tokens(self, n_tokens):
        if hasattr(self.encoder, "set_n_tokens"):
            self.encoder.set_n_tokens(n_tokens)
        if hasattr(self.decoder, "set_n_tokens"):
            self.decoder.set_n_tokens(n_tokens)

    def reset_hidden_state(self):
        if hasattr(self.encoder, "reset_hidden_state"):
            self.encoder.reset_hidden_state()
        if hasattr(self.decoder, "reset_hidden_state"):
            self.decoder.reset_hidden_state()

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
            #     "frequency": 100,
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
            reconstructions, samples, intermediates = ret_values
        else:
            reconstructions, samples = ret_values

        rec = postprocess(reconstructions)
        log = {"reconstructions": [[r] for r in rec]}
        # samples = postprocess(samples)
        # log["samples"] = [[s] for s in samples]
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
