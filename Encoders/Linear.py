import torch
import numpy as np
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from torch import nn
import torch.distributions as tdist
from torch.nn.utils import spectral_norm, remove_spectral_norm
import pytorch_lightning as pl


class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        return y
    

def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)
