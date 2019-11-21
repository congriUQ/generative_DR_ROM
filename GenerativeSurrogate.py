'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
import time
from torch.utils.tensorboard import SummaryWriter
import datetime


class PcNet(nn.Module):
    # mapping from latent z-space to effective diffusivities lambda
    def __init__(self, dim_z, rom_nCells):
        super(PcNet, self).__init__()
        self.fc0 = nn.Linear(dim_z, rom_nCells)

    def forward(self, z):
        lambda_c = self.fc0(z)
        return lambda_c


# class PfNet_old(nn.Module):
#     # From latent z-space to fine scale input data lambda_f
#     def __init__(self, dim_z, dim_x=2):
#         super(PfNet, self).__init__()
#         # dim_out_x == dim_out_z if there're no intermediate layers
#         dim_out_x1 = 4
#         dim_out_z = 4
#         dim_out_s = 2
#         self.fcx0 = nn.Linear(dim_x, dim_out_x1)
#         self.acx0 = nn.ReLU()
#         self.fcz0 = nn.Linear(dim_z, dim_out_z)
#         self.acz0 = nn.ReLU()
#         self.fcs0 = nn.Linear(dim_out_x1 + dim_out_z, dim_out_s)
#         self.acs0 = nn.ReLU()
#         self.fcs1 = nn.Linear(dim_out_s, 1)
#         self.acs1 = nn.Sigmoid()
#
#     def forward(self, z, x):
#         # z.shape == (batchSizeN, imgResolution**2, batchSizeZ, dim_z)
#         # x.shape == (imgResolution**2, 2)
#         out_x = self.fcx0(x)         # out_x.shape = (imgResolution**2, dim_out_x)
#         out_x = self.acx0(out_x)
#         out_z = self.fcz0(z)         # out_z.shape = (batchSizeN, imgResolution**2, batchSizeZ, dim_out_z)
#         out_z = self.acz0(out_z)
#         zs = out_z.shape
#         xs = out_x.shape
#         # Expand coordinate layer output to batchSizeN, batchSizeZ
#         out_x = out_x.unsqueeze(1)
#         out_x = out_x.unsqueeze(0)
#         out_x = out_x.expand(zs[0], xs[0], zs[1], xs[1])
#         # Expand z layer output to pixels
#         out_z = out_z.unsqueeze(1)
#         out_z = out_z.expand(zs[0], xs[0], zs[1], zs[2])
#         out = torch.cat((out_x, out_z), dim=3)  # out.shape = (batchSizeN, imgResolution**2, batchSizeZ, out_x + out_z)
#         out = self.fcs0(out)     # out.shape = (batchSizeN, imgResolution**2, batchSizeZ, 1)
#         out = self.acs0(out)
#         out = self.fcs1(out)
#         out = self.acs1(out)
#         return out


class PfNet(nn.Module):
    # From latent z-space to fine scale input data lambda_f
    def __init__(self, dim_z, dim_img):
        # dim_img = imgResolution**2, i.e., total number of pixels
        super(PfNet, self).__init__()
        self.dim_img = dim_img
        dim_h = int(torch.sqrt(torch.tensor(dim_z*dim_img, dtype=torch.float32)))       # geometric mean
        self.fc0 = nn.Linear(dim_z, dim_h)
        self.ac0 = nn.ReLU()
        self.fc1 = nn.Linear(dim_h, dim_img)
        self.ac1 = nn.Sigmoid()

    def forward(self, z):
        out = self.fc0(z)           # z.shape = (batchSizeN, batchSizeZ, dim_z)
        out = self.ac0(out)
        out = self.fc1(out)
        out = self.ac1(out)         # out.shape = (batchSizeN, batchSizeZ, imgResolution**2)
        return out


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, data, dim_z):
        self.rom = rom
        self.data = data

        self.dim_z = dim_z
        self.z_mean = torch.zeros(self.data.nSamplesIn, dim_z, requires_grad=True)
        # self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))
        self.batchSizeZ = 1             # only point estimates for the time being
        self.varDistOpt = optim.SGD([self.z_mean], lr=3e-3)

        self.pfNet = PfNet(dim_z, self.data.imgResolution**2)
        self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=4e-3)
        self.batchSizeN_pf = min(self.data.nSamplesIn, 256)

        self.pcNet = PcNet(dim_z, rom.mesh.nCells)
        self.pcOpt = optim.Adam(self.pcNet.parameters(), lr=1e-3)
        self.batchSizeN_pc = min(self.data.nSamplesOut, 256)
        self.lambda_c_mean = 3*torch.randn(self.data.nSamplesOut, self.rom.mesh.nCells)

        if __debug__:
            # deletes old log file
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter('runs/gendrrom/' + current_time, flush_secs=5)        # for tensorboard

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_pf(self, pred, batchSamples):
        eps = 1e-16
        return -torch.dot(self.data.microstructImg[batchSamples, :].flatten(),
                          torch.mean(torch.log(pred + eps), dim=1).flatten()) - \
                torch.dot(1 - self.data.microstructImg[batchSamples, :].flatten(),
                          torch.mean(torch.log(1 - pred + eps), dim=1).flatten())

    def loss_pc(self, pred, batchSamples):
        lambda_c_mean = self.lambda_c_mean[batchSamples, :]
        lambda_c_mean = lambda_c_mean.unsqueeze(1)
        lambda_c_mean = lambda_c_mean.expand(pred.shape)
        return torch.dot(torch.mean(lambda_c_mean - pred, dim=1).flatten(),
                         torch.mean(lambda_c_mean - pred, dim=1).flatten())

    def pfStep(self, batchSamples):
        # One training step for pf
        # batchSamples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        # z = torch.randn(self.batchSizeN_pf, self.batchSizeZ, self.dim_z)
        z = self.z_mean[batchSamples, :]
        z = z.unsqueeze(1)
        pred = self.pfNet(z)
        loss = self.loss_pf(pred, batchSamples)
        if __debug__:
            print('loss_pf = ', loss)
            self.writer.add_scalar('Loss/train_pf', loss)
            self.writer.close()
        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()

    def pcStep(self, batchSamples):
        # One training step for pc
        # batchSamples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        # z = torch.randn(self.batchSizeN_pc, self.batchSizeZ, self.dim_z)
        z = self.z_mean[batchSamples, :]
        z = z.unsqueeze(1)
        pred = self.pcNet(z)
        loss = self.loss_pc(pred, batchSamples)
        if __debug__:
            print('loss_pc = ', loss)
            self.writer.add_scalar('Loss/train_pc', loss)
            self.writer.close()
        loss.backward()
        self.pcOpt.step()
        self.pcOpt.zero_grad()

    def neg_log_q_z(self, Z):
        # negative latent log distribution over all z's
        eps = 1e-16
        pred_c = self.pcNet(Z[:self.data.nSamplesOut, :])
        # precision of pc still needs to be added!!
        out = .5*torch.dot((self.lambda_c_mean - pred_c).flatten(), (self.lambda_c_mean - pred_c).flatten())
        pred_f = self.pfNet(Z)
        out -= torch.dot(self.data.microstructImg.flatten(), torch.log(pred_f + eps).flatten()) + \
               torch.dot(1 - self.data.microstructImg.flatten(), torch.log(1 - pred_f + eps).flatten())
        return out

    def optLatentDistStep(self):
        # optimize latent distribution p(lambda_c^n, z^n) for point estimates
        self.lambda_c_mean = self.pcNet(self.z_mean[:self.data.nSamplesOut, :])

        # Z = self.z_mean.clone().detach().requires_grad_(True)
        # optimizer_z = optim.SGD([Z], lr=3e-2)
        for k in range(5):
            loss_z = self.neg_log_q_z(self.z_mean)
            loss_z.backward(retain_graph=True)
            self.varDistOpt.step()
            self.varDistOpt.zero_grad()
        # self.z_mean = Z.detach()

    def plotInputReconstruction(self):
        fig, ax = plt.subplots(1, 5)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # This needs to be replaced by the (approximate) posterior on z!!
        nSamplesZ = 100
        z = torch.randn(1, nSamplesZ, self.dim_z)
        samples = self.pfNet(z)
        # pred_mean = torch.mean(samples, dim=1)
        mpbl = ax[0].imshow(torch.reshape(samples[0, 0, :],
                                   (self.data.imgResolution, self.data.imgResolution)).detach().numpy())
        ax[0].set_xticks([], [])
        ax[0].set_yticks([], [])
        pos = ax[0].get_position()
        cb_ax = cb.make_axes(ax[0], location='left', shrink=.35, anchor=(-5.0, .5), ticklocation='left')
        cbr = plt.colorbar(mpbl, cax=cb_ax[0])
        # fig.colorbar(mpbl, ax=ax[0], location='left', pad=0.0)
        ax[0].set_title('p(lambda| z_0)')
        ax[0].set_position(pos)
        ax[1].imshow(torch.reshape(samples[0, 0, :],
                                   (self.data.imgResolution, self.data.imgResolution)).detach().numpy() > .5,
                     cmap='binary')
        ax[1].set_xticks([], [])
        ax[1].set_yticks([], [])
        ax[1].set_title('p(lambda| z_0) > .5')
        ax[2].imshow((torch.reshape(samples[0, 1, :], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[2].set_xticks([], [])
        ax[2].set_yticks([], [])
        ax[2].set_title('sample')
        ax[3].imshow((torch.reshape(samples[0, 2, :], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[3].set_xticks([], [])
        ax[3].set_yticks([], [])
        ax[3].set_title('sample')
        ax[4].imshow((torch.reshape(samples[0, 3, :], (self.data.imgResolution, self.data.imgResolution)) >
                      torch.rand(self.data.imgResolution, self.data.imgResolution)).detach().numpy(),
                     cmap='binary')
        ax[4].set_xticks([], [])
        ax[4].set_yticks([], [])
        ax[4].set_title('sample')














