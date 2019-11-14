'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np


class Pc(nn.Module):
    # mapping from latent z-space to effective diffusivities lambda
    def __init__(self, dim_z, rom_nCells):
        super(Pc, self).__init__()
        self.fc0 = nn.Linear(dim_z, rom_nCells)

    def forward(self, z):
        lambda_c = torch.exp(self.fc0(z))
        return lambda_c


class Pf(nn.Module):
    # From latent z-space to fine scale input data lambda_f
    def __init__(self, dim_z, resolution):
        super(Pf, self).__init__()
        self.fc0 = nn.Linear(dim_z, resolution**2)

    def forward(self, z):
        return self.fc0(z)


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, pc, dim_z):
        self.rom = rom
        self.pc = pc
        self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass













