from vle.modules.functional import epsilon
import lpips

import torch.nn.functional as F
import torch.nn as nn
import torch


def normalize_token_along_batch(t):
    return (t - torch.mean(t, dim=0)) / (torch.std(t, dim=0) + epsilon)


def get_cross_correlation_loss(t1, t2, lambd=1e-3):
    (b, c, h, w) = t1.shape
    d = c * h * w

    # Flatten channel, height, width dimensions and normalize
    t1 = normalize_token_along_batch(t1.reshape(b, d))  # b x c * h * w
    t2 = normalize_token_along_batch(t2.reshape(b, d))  # b x c * h * w

    # Calculate cross correlation within each batch
    cc = 1 - torch.abs(torch.sum(t1 * t2, dim=1) / d)  # b

    return lambd / torch.mean(cc)


# https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
def kld_loss(mu, log_var, other_mu=None, other_log_var=None):
    mu = mu.flatten(start_dim=1)
    log_var = log_var.flatten(start_dim=1)
    # Assert that both are None or both are not None
    assert (other_mu is None and other_log_var is None) or (
        other_mu is not None and other_log_var is not None
    )
    if other_mu is None:
        return torch.mean(
            -0.5
            * torch.sum(
                1 + log_var - mu**2 - log_var.exp(),
                dim=1,
            ),
            dim=0,
        )
    other_mu = other_mu.flatten(start_dim=1)
    other_log_var = other_log_var.flatten(start_dim=1)
    return torch.mean(
        torch.sum(
            0.5
            * (
                -1.0
                + other_log_var
                - log_var
                + torch.exp(log_var - other_log_var)
                + ((mu - other_mu) ** 2) * torch.exp(-other_log_var)
            ),
            dim=1,
        ),
        dim=0,
    )


def masks_diff_loss(x, y, scale=5, mask=None):
    error = torch.square(x - y)
    if mask is None:
        error = error.mean()
        return torch.exp(-scale * error)
    loss = torch.exp(-scale * error)
    return (mask * loss).mean()


def total_variation_loss(x, stride=1):
    assert stride < x.size(-1) and stride < x.size(-2)
    shifted_x = x[..., :-stride, :-stride]
    h_loss = F.mse_loss(shifted_x, x[..., :-stride, stride:])
    v_loss = F.mse_loss(shifted_x, x[..., stride:, :-stride])
    return (h_loss + v_loss) / 2


def mse_with_mask(x, y, mask):
    squared_error = (x - y) ** 2
    return torch.mean(squared_error * mask)


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lpips_loss = lpips.LPIPS(net="alex")
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        self.lpips_loss = self.lpips_loss.eval()

    def forward(self, x, y):
        return self.lpips_loss(x, y).mean()


class SoftDiceLossV1(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super().__init__()

        self.smooth = smooth
        self.p = p

    def forward(self, prediction, labels):
        """
        inputs:
            prediction: tensor of shape (N, C, H, W, ...)
            labels: tensor of shape(N, C, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        numer = (prediction * labels).sum()
        denor = (prediction.pow(self.p) + labels.pow(self.p)).sum()
        return 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
