from vle.utils import instantiate_from_config
from vle.modules.loss import LPIPSLoss
from vle.modules.functional import (
    epsilon,
    folded_normal_pdf,
    inverse_input_norm, 
    postprocess, 
    normalize,
)

import pytorch_lightning as pl
import torch.nn.functional as F
import torch

import numpy as np


def lr_lambda_fn(epoch):
    return 0.65**epoch


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        max_tokens=10,
        update_token_dist_step=5000,
        learning_rate=1e-3,
        use_p_loss=True,
        aux_mult=1.0,
        p_mult=0.1,
    ):
        super().__init__()
        self.enc = instantiate_from_config(encoder)
        self.dec = instantiate_from_config(decoder)

        self.p_mult = p_mult
        self.use_p_loss = use_p_loss
        self.aux_mult = aux_mult
        self.lr = learning_rate

        self.lpips_loss = LPIPSLoss() if use_p_loss else None

        self.max_tokens = max_tokens
        self.tokens_range = np.arange(1, max_tokens + 1)
        # Distribution to follow for selecting `n_tokens`
        self.token_choice_dist = folded_normal_pdf
        # Update distribution for token sampling after `update_token_dist_step`
        self.update_token_dist_step = update_token_dist_step

        self.downsample_factor = 2 ** (len(encoder["params"]["channel_mults"]) - 1)
        self.latent_channels = encoder["params"]["latent_channels"]

    def load_weights(self, ckpt_path, verbose=False):
        state_dict = torch.load(ckpt_path)
        sd = state_dict["state_dict"]
        keys_to_remove = []
        for key in sd:
            if key.startswith("lpips_loss"):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del sd[key]
        state_dict["state_dict"] = sd
        self.load_state_dict(state_dict["state_dict"])
        if verbose:
            print("Weights sucessfully loaded!")

    def training_step(self, batch, batch_idx):
        # bs, ch, height, width = inputs.shape
        x = batch
        n_tokens = self.sample_n_tokens()
        _, loss_dict = self(x, n_tokens, compute_losses=True)

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
            ep = min(self.global_step / self.update_token_dist_step, 0.5)
            if loss_name == "aux_loss":
                loss_value *= 1 - ep
            if loss_name == "rec_loss":
                loss_value *= ep
            if loss_name == "lpips_loss":
                loss_value *= self.p_mult
            total_loss += loss_value

        return total_loss

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

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def _forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def forward(self, x, n_tokens, compute_losses=False, log=False, postprocess=True):
        if log:
            tokens = []
        if compute_losses:
            aux_rec_loss = 0.0

        rec = torch.zeros_like(x)
        for _ in range(n_tokens):
            model_input = x - rec
            residuals, mean, std = normalize(x - rec)
            i_rec = self._forward(residuals) * std + mean
            rec = rec + i_rec

            if compute_losses:
                aux_rec_loss += F.mse_loss(model_input, i_rec)

            if log:
                if postprocess:
                    i_rec = inverse_input_norm(i_rec.clamp(-1.0, 1.0))
                tokens.append(i_rec)

        if compute_losses:
            p_loss = 0.0
            if self.lpips_loss is not None:
                p_loss = self.lpips_loss(x, rec).mean()

            rec_loss = F.mse_loss(x, rec)
            aux_rec_loss = aux_rec_loss / n_tokens

            loss_dict = {
                "aux_loss": aux_rec_loss,
                "rec_loss": rec_loss,
                "lpips_loss": p_loss,
            }
            return rec, loss_dict

        if log:
            intermediates = {
                "model_output": tokens,
            }
            rec = inverse_input_norm(rec.clamp(-1.0, 1.0))
            return rec, intermediates

        return rec

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    @torch.no_grad()
    def log_img(self, batch, batch_idx, sample_intermediates=False, **kwargs):
        ret_values = self(
            batch,
            self.max_tokens,
            log=sample_intermediates,
        )
        if sample_intermediates:
            reconstructions, intermediates = ret_values
        else:
            reconstructions = ret_values

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
