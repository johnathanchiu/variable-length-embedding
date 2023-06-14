from vle.deprecated.simple import Encoder, Decoder, TokenRNN 
from vle.deprecated.resnet import (
    ResNet, 
    ResNetTranspose,
    Bottleneck, 
    BottleneckTranspose,
)

import pytorch_lightning as pl
import torch.nn.functional as F
import torch


def postprocess(x):
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = x.moveaxis(1, -1) * 255
    if x.size(-1) == 1:
        x = x.squeeze(-1)
    return x.detach().cpu().numpy()


class TokenModel(pl.LightningModule):

    token_dim = 128
    encode_dim = 128

    def __init__(self, simple_experiment=False, num_tokens=10, lr=1e-3):
        super().__init__()        

        if simple_experiment:
            self.enc = Encoder()
            self.tok = TokenRNN(64, 128)
            self.dec = Decoder(64, 3200)
        else:
            self.enc = ResNet(Bottleneck, [3, 4, 6, 3], encode_dim=self.encode_dim, in_planes=32)
            self.tok = TokenRNN(self.token_dim, self.encode_dim)
            self.dec = ResNetTranspose(BottleneckTranspose, [3, 6, 4, 3], in_planes=self.token_dim)

        self.num_tokens = num_tokens
        self.lr = lr

    def load_weights(self, ckpt_path, verbose=False):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict["state_dict"])
        if verbose:
            print("Weights sucessfully loaded!")

    def forward(self, x, compute_losses=False, log=False):
        if log:
            tokens = []

        rec = torch.zeros_like(x) # Reconstruction starts at all zeros    
        z = self.enc(x) # Encode image into some latent space (fixed dim)
        y, hid_state = self.tok.initialize_inputs(z) # Start generating tokens with LSTM. 
        for _ in range(self.num_tokens): # FIXED: 10 tokens per image
            y, hid_state = self.tok(z, y, hid_state) # Get next token from LSTM 
            i_rec = self.dec(y) # Decode using token, add to reconstruction
            rec = rec + i_rec

            if log:
                tokens.append(i_rec)

        if compute_losses:
            loss_dict = {
                "mse_loss": F.mse_loss(x, rec),
            }
            return rec, loss_dict
        
        if log:
            log_dict = {
                "model_output": tokens, 
            }
            return rec, log_dict

        return rec

    def training_step(self, batch, batch_idx):
        y, loss_dict = self(batch, compute_losses=True)

        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            self.log(loss_name, loss_value, on_step=True, on_epoch=False, prog_bar=True)
            total_loss += loss_value

        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
    
    @torch.no_grad()
    def log_img(self, batch, batch_idx, sample_intermediates=False, **kwargs):
        ret_values = self(
            batch,
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