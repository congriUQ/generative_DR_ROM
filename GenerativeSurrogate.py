'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import optim
import numpy as np


class Pc(nn.Module):
    # mapping from latent z-space to effective diffusivities lambda
    def __init__(self, dim_z, rom_nCells):
        super(Pc, self).__init__()
        self.fc0 = nn.Linear(dim_z, rom_nCells)

    def forward(self, z):
        lambda_c = torch.exp(self.fc0(z))
        return lambda_c


class PfNet(nn.Module):
    # From latent z-space to fine scale input data lambda_f
    def __init__(self, dim_z, dim_x=2):
        super(PfNet, self).__init__()
        self.fc0 = nn.Linear(dim_x + dim_z, 1)
        self.currLoss = None

    def forward(self, zx):
        return torch.sigmoid(self.fc0(zx))


# class Pf:
#     def __init__(self, net, data):
#         self.net = net
#         self.data = data
#         self.opt = optim.Adam(self.net.parameters(), lr=.001)
#
#     def lossFun(self, pred_out):
#         # out data of p_f is lambda_f
#         return torch.dot(pred_out, data_out) + torch.dot((1.0 - pred_out), (1.0 - data_out))
#
#     def trainStep(self):





class GenerativeSurrogate:
    # model class
    def __init__(self, rom, data, dim_z):
        self.rom = rom
        self.data = data
        self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))
        self.pfNet = PfNet(dim_z)
        self.pcNet = Pc(dim_z, rom.mesh.nCells)

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_pf(self, predOut):
        return torch.dot(predOut, inp) + torch.dot((1.0 - predOut), (1.0 - inp))













