'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
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
        # dim_out_x == dim_out_z if there're no intermediate layers
        dim_out_x = 10
        dim_out_z = 4
        self.fcx = nn.Linear(dim_x, dim_out_x)
        self.fcz = nn.Linear(dim_z, dim_out_z)
        self.fcs = nn.Linear(dim_out_x + dim_out_z, 1)

    def forward(self, z, x):
        # z.shape == (batchSizeN, imgResolution**2, batchSizeZ, dim_z)
        # x.shape == (imgResolution**2, 2)
        out_x = self.fcx(x)         # out_x.shape = (imgResolution**2, dim_out_x)
        out_x = torch.sigmoid(out_x)
        # Expand coordinate layer output to batchSizeN
        out_z = self.fcz(z)         # out_z.shape = (batchSizeN, imgResolution**2, batchSizeZ, dim_out_z)
        out_z = torch.sigmoid(out_z)
        zs = out_z.shape
        xs = out_x.shape
        out_x = out_x.unsqueeze(1)
        out_x = out_x.unsqueeze(0)
        out_x = out_x.expand(zs[0], xs[0], zs[1], xs[1])
        out_z = out_z.unsqueeze(1)
        out_z = out_z.expand(zs[0], xs[0], zs[1], zs[2])
        out = torch.cat((out_x, out_z), dim=3)  # out.shape = (batchSizeN, imgResolution**2, batchSizeZ, out_x + out_z)
        out = self.fcs(out)     # out.shape = (batchSizeN, imgResolution**2, batchSizeZ, 1)
        out = torch.sigmoid(out)
        return out


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, data, dim_z):
        self.rom = rom
        self.data = data
        self.dim_z = dim_z
        self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))
        self.batchSizeN = min(self.data.nSamples, 128)
        self.batchSizeZ = 10
        self.pfNet = PfNet(dim_z)
        self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=3e-3)
        self.pcNet = Pc(dim_z, rom.mesh.nCells)
        if __debug__:
            # deletes old log file
            self.log_pf_loss = open('./log_pf_loss.txt', 'w+')
            self.log_pf_loss.close()

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_pf(self, predOut, batchSamples):
        return -torch.dot(self.data.microstructImg[batchSamples, :].flatten(),
                          torch.mean(torch.log(predOut), dim=2).flatten()) - \
                torch.dot(1 - self.data.microstructImg[batchSamples, :].flatten(),
                          torch.mean(torch.log(1 - predOut), dim=2).flatten())

    def pfStep(self, batchSamples):
        # One training step
        # batchSamples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = torch.randn(self.batchSizeN, self.batchSizeZ, self.dim_z)
        # print('z0 = ', z.nelement())
        # # Same sample for z is valid in every pixel of the image
        # z = z.expand(self.batchSizeN, self.data.imgResolution**2, self.batchSizeZ, self.dim_z)
        # print('z1 = ', z.nelement())
        pred = self.pfNet(z, self.data.imgX)
        loss = self.loss_pf(pred, batchSamples)
        if __debug__:
            print('loss = ', loss)
            self.log_pf_loss = open('./log_pf_loss.txt', 'a+')
            self.log_pf_loss.write(str(loss.detach().numpy()) + '\n')
            self.log_pf_loss.close()

        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()

    def plotInputReconstruction(self):
        fig, ax = plt.subplots(1, 5)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # This needs to be replaced by the (approximate) posterior on z!!
        nSamplesZ = 100
        z = torch.randn(1, nSamplesZ, self.dim_z)
        samples = self.pfNet(z, self.data.imgX)
        pred_mean = torch.mean(samples, dim=2)
        mpbl = ax[0].imshow(torch.reshape(samples[:, :, 0],
                                   (self.data.imgResolution, self.data.imgResolution)).detach().numpy())
        ax[0].set_xticks([], [])
        ax[0].set_yticks([], [])
        pos = ax[0].get_position()
        cb_ax = cb.make_axes(ax[0], location='left', shrink=.35, anchor=(-5.0, .5), ticklocation='left')
        cbr = plt.colorbar(mpbl, cax=cb_ax[0])
        # fig.colorbar(mpbl, ax=ax[0], location='left', pad=0.0)
        ax[0].set_title('p(lambda| z_1)')
        ax[0].set_position(pos)
        ax[1].imshow(torch.reshape(pred_mean,
                                   (self.data.imgResolution, self.data.imgResolution)).detach().numpy() > .5,
                     cmap='binary')
        ax[1].set_xticks([], [])
        ax[1].set_yticks([], [])
        ax[1].set_title('p(lambda) > .5')
        ax[2].imshow((torch.reshape(samples[:, :, 0], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[2].set_xticks([], [])
        ax[2].set_yticks([], [])
        ax[2].set_title('sample')
        ax[3].imshow((torch.reshape(samples[:, :, 1], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[3].set_xticks([], [])
        ax[3].set_yticks([], [])
        ax[3].set_title('sample')
        ax[4].imshow((torch.reshape(samples[:, :, 2], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[4].set_xticks([], [])
        ax[4].set_yticks([], [])
        ax[4].set_title('sample')














