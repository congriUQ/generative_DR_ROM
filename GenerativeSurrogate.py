'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import optim
import numpy as np
import time


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
        out_x = 1
        out_z = 1
        self.fcx = nn.Linear(dim_x, out_x)
        self.fcz = nn.Linear(dim_z, out_z)
        self.fcs = nn.Linear(out_x + out_z, 1)

    def forward(self, z, x):
        out_x = self.fcx(x)
        # Expand coordinate layer output to batchSizeN
        out_z = self.fcz(z)
        zs = out_z.shape
        out_x = out_x.unsqueeze(1)
        out_x = out_x.unsqueeze(0)
        out_x = out_x.expand(zs[0], zs[1], zs[2], zs[3])
        out = torch.cat((out_x, out_z), dim=3)
        out = self.fcs(out)
        out = torch.sigmoid(out)
        return out


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, data, dim_z):
        self.rom = rom
        self.data = data
        self.dim_z = dim_z
        self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))
        self.batchSizeZ = 10
        self.pfNet = PfNet(dim_z)
        self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=1e-3)
        self.pcNet = Pc(dim_z, rom.mesh.nCells)

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_pf(self, predOut):
        return -torch.dot(self.data.microstructImg.flatten(), torch.mean(torch.log(predOut), dim=2).flatten()) - \
                torch.dot(1 - self.data.microstructImg.flatten(), torch.mean(torch.log(1 - predOut), dim=2).flatten())

    def pfStep(self):
        # One training iteration
        # This needs to be replaced by the (approximate) posterior on z!!
        z = torch.randn(self.data.nSamples, self.data.imgResolution**2, self.batchSizeZ, self.dim_z)
        pred = self.pfNet(z, self.data.imgX)
        loss = self.loss_pf(pred)
        print('loss = ', loss)
        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()














