from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from torch import nn
import torch.distributions as tdist
from torch.nn.utils import spectral_norm, remove_spectral_norm


class BatchedNormalizingFlowDensity(nn.Module):

    def __init__(self, c, dim, flow_length, flow_type='planar_flow'):
        super(BatchedNormalizingFlowDensity, self).__init__()
        self.c = c
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.c, self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim).repeat(self.c, 1, 1), requires_grad=False)

        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(c, dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(
            self.mean.repeat(z.size(1), 1, 1).permute(1, 0, 2),
            self.cov.repeat(z.size(1), 1, 1, 1).permute(1, 0, 2, 3)
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x
