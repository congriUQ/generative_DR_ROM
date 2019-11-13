'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as fnl
import torch.distributions as dist
import numpy as np


# class ProbabilityDistribution:
#     def __init__(self, params=None):
#         self.params = params
#
#
# class Gaussian(ProbabilityDistribution):
#     def __init__(self, mu=.0, sigma=1.0):
#         super().__init__(params={mu: mu, sigma: sigma})
#         pass


class PC(nn.Module):
    # mapping from latent z-space to effective diffusivities lambda

    def __init__(self, z_dim, rom_nEq):
        super(PC, self).__init__()
        self.fc0 = nn.Linear(z_dim, rom_nEq)

    def forward(self, x):
        x = torch.exp(self.fc0(x))
        return x


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, pc, dim_z):
        self.rom = rom
        self.pc = pc
        self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))

    def train(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass









