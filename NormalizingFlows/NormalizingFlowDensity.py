from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from torch import nn
import torch.distributions as tdist
from torch.nn.utils import spectral_norm, remove_spectral_norm


class NormalizingFlowDensity(nn.Module):

    def __init__(self, dim, flow_length, flow_type='planar_flow', ROI_level=False, Attention=True):
        super(NormalizingFlowDensity, self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type
        self.ROI_level = ROI_level
        self.Attention = Attention
        attention_heads = 1

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        self.attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=attention_heads, batch_first=True)

        # self.attention2 = nn.MultiheadAttention(embed_dim=10, num_heads=attention_heads, batch_first=True)

        self.labmbda1 = torch.tensor(0.)
        self.labmbda2 = torch.tensor(1.)

        self.batchnorm1 = nn.BatchNorm1d(num_features=10)
        self.batchnorm2 = nn.BatchNorm1d(num_features=10)


        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            self.transforms = nn.Sequential(*(
                affine_autoregressive(dim, hidden_dims=[1024, 1024]) for _ in range(flow_length)
            ))
        else:
            raise NotImplementedError

    def forward(self, z):

        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):

        if self.ROI_level:
            if self.Attention:
                batch_size, num_of_ROI, _ = x.shape
                x_global = self.batchnorm1(torch.mean(x, dim=1))

                x_batch_norm = self.batchnorm1(x.transpose(1,2)).transpose(1,2)
                
                attn_output, _ = self.attention(query = x_global.unsqueeze(1), key = x_batch_norm, value = x_batch_norm)

                z, sum_log_jacobians = self.forward(attn_output.squeeze(1))
                log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
                log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
                return log_prob_x

            else:
                batch_size, num_of_ROI, _ = x.shape
                x_global = self.batchnorm1(torch.mean(x, dim=1))
                

                # atten_x,_ = self.attention2(x, x, x)
                # atten_x = self.batchnorm1(torch.mean(atten_x, dim=1))

                # Flatten ROIs for individual density computation
                z_flat, sum_log_jacobians = self.forward(self.batchnorm2(x.view(-1, self.dim)))
                z_flat = torch.nan_to_num(z_flat, nan=0.0)
                log_prob_z_flat = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z_flat)
                log_prob_x_flat = log_prob_z_flat + sum_log_jacobians
                log_prob_x = log_prob_x_flat.view(batch_size, num_of_ROI)

                # log_prob_x = torch.mean(log_prob_x, dim=1)
                # Global density computation
                
                z_global, sum_log_jacobians_global = self.forward(x_global) # x_global
                z_global = torch.nan_to_num(z_global, nan=0.0)
                log_prob_z_global = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z_global)
                log_prob_x_global = log_prob_z_global + sum_log_jacobians_global
                global_context_density = log_prob_x_global.unsqueeze(1).expand(-1, x.shape[1])

                # Use the global_context_density as the query for attention
                log_prob_x = log_prob_x.unsqueeze(-1)
                global_context_density = global_context_density.unsqueeze(-1)
                attn_output, attn_weights = self.attention(global_context_density, log_prob_x, log_prob_x)
                attn_output = attn_output.squeeze(-1)

                # return self.labmbda1 * torch.mean(log_prob_x.squeeze(-1),dim=1) + self.labmbda2 * log_prob_x_global
                return self.labmbda1 * torch.mean(attn_output,dim=1) + self.labmbda2 * log_prob_x_global
                # return torch.mean(self.labmbda1*attn_output + self.labmbda2*global_context_density.squeeze(-1), dim=1)
                # return log_prob_x_global # -9
                # return log_prob_x
                # return torch.mean(log_prob_x.squeeze(-1),dim=1)
        
        
        else:
            z, sum_log_jacobians = self.forward(x)
            log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
            log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
            return log_prob_x
