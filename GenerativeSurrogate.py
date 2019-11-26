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
import numpy as np
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


class LogPcf(torch.autograd.Function):
    """This implements the Darcy ROM as a torch autograd function"""

    @staticmethod
    def forward(ctx, input):
        # X is typically the log diffusivity, i.e., X = log(lambda_c)
        out = input**2
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        print('input = ', input)
        grad_input = grad_output.clone()
        grad_input = 2*input*grad_input
        return grad_input


class PcfNet(nn.Module):
    # mapping from coarse solution u_c back to fine solution u_f
    def __init__(self, W_cf):
        super(PcfNet, self).__init__()
        # W_cf is a PETSc matrix!
        self.fc0 = W_cf

    def forward(self):
        pass


class PcNet(nn.Module):
    # mapping from latent z-space to effective diffusivities lambda
    def __init__(self, dim_z, rom_n_cells):
        super(PcNet, self).__init__()
        self.fc0 = nn.Linear(dim_z, rom_n_cells)
        self.ac0 = nn.Softplus()

    def forward(self, z):
        lambda_c = self.fc0(z)
        # for positiveness -- exp is not necessary!
        lambda_c = self.ac0(lambda_c)
        return lambda_c


class PfNet(nn.Module):
    # From latent z-space to fine scale input data lambda_f
    def __init__(self, dim_z, dim_img):
        # dim_img = img_resolution**2, i.e., total number of pixels
        super(PfNet, self).__init__()
        self.dim_img = dim_img
        dim_h = int(torch.sqrt(torch.tensor(dim_z*dim_img, dtype=torch.float32)))       # geometric mean
        self.fc0 = nn.Linear(dim_z, dim_h)
        self.ac0 = nn.ReLU()
        self.fc1 = nn.Linear(dim_h, dim_img)
        self.ac1 = nn.Sigmoid()

    def forward(self, z):
        out = self.fc0(z)           # z.shape = (batch_size_N, batchSizeZ, dim_z)
        out = self.ac0(out)
        out = self.fc1(out)
        out = self.ac1(out)         # out.shape = (batch_size_N, batchSizeZ, img_resolution**2)
        return out


class GenerativeSurrogate:
    # model class
    def __init__(self, rom, data, dim_z):
        self.dtype = torch.float32

        self.rom = rom
        self.data = data

        self.dim_z = dim_z
        self.z_mean = torch.zeros(self.data.n_samples_in, dim_z, requires_grad=True)
        # self.pz = dist.Normal(torch.tensor(dim_z*[.0]), torch.tensor(dim_z*[1.0]))
        self.batch_size_z = 1             # only point estimates for the time being
        self.varDistOpt = optim.SGD([self.z_mean], lr=1e-3)

        self.pfNet = PfNet(dim_z, self.data.img_resolution**2)
        self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=4e-3)
        self.batch_size_N_pf = min(self.data.n_samples_in, 1024)

        self.pcNet = PcNet(dim_z, rom.mesh.n_cells)
        self.pcOpt = optim.Adam(self.pcNet.parameters(), lr=1e-3)
        self.batch_size_N_pc = min(self.data.n_samples_out, 256)
        self.lambda_c_mean = 3*torch.randn(self.data.n_samples_out, self.rom.mesh.n_cells)

        # Change for non unit square domains!!
        xx, yy = np.meshgrid(np.linspace(0, 1, self.data.output_resolution),
                             np.linspace(0, 1, self.data.output_resolution))
        X = np.concatenate((np.expand_dims(xx.flatten(), axis=1), np.expand_dims(yy.flatten(), axis=1)), axis=1)
        self.rom.mesh.get_interpolation_matrix(X)
        self.pcfNet = PcfNet(self.rom.mesh.interpolation_matrix)

        if __debug__:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter('runs/gendrrom/' + current_time, flush_secs=5)        # for tensorboard

    def fit(self):
        # method to train the model
        pass

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_pf(self, pred, batch_samples):
        eps = 1e-16
        return -torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(pred + eps), dim=1).flatten()) - \
                torch.dot(1 - self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(1 - pred + eps), dim=1).flatten())

    def loss_pc(self, pred, batch_samples):
        # needs to be updated to samples of lambda_c!!
        lambda_c_mean = self.lambda_c_mean[batch_samples, :]
        lambda_c_mean = lambda_c_mean.unsqueeze(1)
        lambda_c_mean = lambda_c_mean.expand(pred.shape)
        return torch.dot(torch.mean(lambda_c_mean - pred, dim=1).flatten(),
                         torch.mean(lambda_c_mean - pred, dim=1).flatten())

    def loss_pcf(self, pred, batch_samples):

        return None

    def pf_step(self, batch_samples):
        # One training step for pf
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        # z = torch.randn(self.batch_size_N_pf, self.batch_size_z, self.dim_z)
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)
        pred = self.pfNet(z)
        loss = self.loss_pf(pred, batch_samples)
        if __debug__:
            print('loss_pf = ', loss)
            self.writer.add_scalar('Loss/train_pf', loss)
            self.writer.close()
        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()

    def pc_step(self, batch_samples):
        # One training step for pc
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        # z = torch.randn(self.batch_size_N_pc, self.batch_size_z, self.dim_z)
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)
        pred = self.pcNet(z)
        loss = self.loss_pc(pred, batch_samples)
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
        pred_c = self.pcNet(Z[:self.data.n_samples_out, :])
        # precision of pc still needs to be added!!
        # out = .5*torch.dot((self.lambda_c_mean - pred_c).flatten(), (self.lambda_c_mean - pred_c).flatten())
        pred_f = self.pfNet(Z)
        # out -= ... !!
        out = -(torch.dot(self.data.microstructure_image.flatten(), torch.log(pred_f + eps).flatten()) + \
               torch.dot(1 - self.data.microstructure_image.flatten(), torch.log(1 - pred_f + eps).flatten())) + \
                .5*torch.sum(Z*Z)
        return out

    def opt_latent_dist_step(self):
        # optimize latent distribution p(lambda_c^n, z^n) for point estimates
        self.lambda_c_mean = self.pcNet(self.z_mean[:self.data.n_samples_out, :])

        # Z = self.z_mean.clone().detach().requires_grad_(True)
        # optimizer_z = optim.SGD([Z], lr=3e-2)
        for k in range(1):
            loss_z = self.neg_log_q_z(self.z_mean)
            loss_z.backward(retain_graph=True)
            self.varDistOpt.step()
            self.varDistOpt.zero_grad()
        # self.z_mean = Z.detach()

    def plot_input_reconstruction(self):
        fig, ax = plt.subplots(1, 5)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # This needs to be replaced by the (approximate) posterior on z!!
        n_samples_z = 100
        z = torch.randn(1, n_samples_z, self.dim_z)
        samples = self.pfNet(z)
        # pred_mean = torch.mean(samples, dim=1)
        mpbl = ax[0].imshow(torch.reshape(samples[0, 0, :],
                                   (self.data.img_resolution, self.data.img_resolution)).detach().numpy())
        ax[0].set_xticks([], [])
        ax[0].set_yticks([], [])
        pos = ax[0].get_position()
        cb_ax = cb.make_axes(ax[0], location='left', shrink=.35, anchor=(-5.0, .5), ticklocation='left')
        cbr = plt.colorbar(mpbl, cax=cb_ax[0])
        # fig.colorbar(mpbl, ax=ax[0], location='left', pad=0.0)
        ax[0].set_title('p(lambda| z_0)')
        ax[0].set_position(pos)
        ax[1].imshow(torch.reshape(samples[0, 0, :],
                                   (self.data.img_resolution, self.data.img_resolution)).detach().numpy() > .5,
                     cmap='binary')
        ax[1].set_xticks([], [])
        ax[1].set_yticks([], [])
        ax[1].set_title('p(lambda| z_0) > .5')
        ax[2].imshow((torch.reshape(samples[0, 1, :], (self.data.img_resolution, self.data.img_resolution)) >
                      torch.rand(self.data.img_resolution, self.data.img_resolution)).detach().numpy(),
                     cmap='binary')
        ax[2].set_xticks([], [])
        ax[2].set_yticks([], [])
        ax[2].set_title('sample')
        ax[3].imshow((torch.reshape(samples[0, 2, :], (self.data.img_resolution, self.data.img_resolution)) >
                      torch.rand(self.data.img_resolution, self.data.img_resolution)).detach().numpy(),
                     cmap='binary')
        ax[3].set_xticks([], [])
        ax[3].set_yticks([], [])
        ax[3].set_title('sample')
        ax[4].imshow((torch.reshape(samples[0, 3, :], (self.data.img_resolution, self.data.img_resolution)) >
                      torch.rand(self.data.img_resolution, self.data.img_resolution)).detach().numpy(),
                     cmap='binary')
        ax[4].set_xticks([], [])
        ax[4].set_yticks([], [])
        ax[4].set_title('sample')














