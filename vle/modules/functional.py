import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

epsilon = 1e-4


class RoundFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class RoundLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return RoundFunction.apply(x)
    

class ClampFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(0.0, 1.0)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class ClampLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ClampFunction.apply(x)


def scaled_tanh(x, scale=3):
    return torch.tanh(scale * x)


def scaled_sigmoid(x, scale=100):
    return torch.sigmoid(scale * x)


def normalize(x):
    mean = torch.mean(x, dim=(-1, -2), keepdims=True)
    std = torch.std(x, dim=(-1, -2), keepdims=True) + epsilon
    return (x - mean) / std, mean, std


def inverse_input_norm(img):
    return (img + 1.0) / 2.0


def input_norm(img):
    return img * 2.0 - 1.0


def postprocess(img):
    img = img.moveaxis(1, -1)
    if img.size(-1) == 1:
        img = img.squeeze(-1)
    return img.detach().cpu().numpy() * 255


def folded_normal_pdf(x, mu=0.0, sigma=1.0):
    sig_squared = sigma**2
    left_half = 1 / np.sqrt(2 * np.pi * sig_squared)
    left_half = left_half * np.exp(-1 * (x - mu) ** 2 / (2 * sig_squared))

    right_half = 1 / np.sqrt(2 * np.pi * sig_squared)
    right_half = right_half * np.exp(-1 * (x + mu) ** 2 / (2 * sig_squared))

    return left_half + right_half
