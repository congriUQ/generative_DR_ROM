'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import myutil as my
import ROM
import Data as dta


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
        # self.ac0 = nn.Softplus(beta=1e-1)

    def forward(self, z):
        eps = 1e-8
        x = self.fc0(z)
        # for positiveness -- exp is not necessary!
        # lambda_c = self.ac0(lambda_c) + eps
        # lambda_c = torch.exp(lambda_c) + eps
        return x


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
    def __init__(self, rom=None, data=None):
        # rom is only allowed to be None if the model is loaded afterwards
        if rom is not None:
            self.dtype = torch.float32

            self.rom = rom
            self.rom_autograd = rom.get_autograd_fun() if rom is not None else None

            if data is None:
                self.data = dta.StokesData()
            else:
                self.data = data

            self.dim_z = 20
            self.z_mean = torch.zeros(self.data.n_supervised_samples + self.data.n_unsupervised_samples,
                                      self.dim_z, requires_grad=True)
            self.lr_z = 1e-3
            self.zOpt = optim.SGD([self.z_mean], lr=self.lr_z)
            self.batch_size_z = min(self.data.n_unsupervised_samples, 128)

            self.pfNet = PfNet(self.dim_z, self.data.img_resolution**2)
            self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=4e-3)
            self.batch_size_N_thetaf = min(self.data.n_unsupervised_samples, 128)

            # so far no batched evaluation implemented. EXTEND THIS!!
            self.batch_size_N_lambdac = min(self.data.n_supervised_samples, 1)
            self.batch_size_N_thetac = min(self.data.n_supervised_samples, 256)

            self.pcNet = PcNet(self.dim_z, rom.mesh.n_cells)
            self.pcOpt = optim.Adam(self.pcNet.parameters(), lr=3e-3)
            self.tauc = torch.ones(self.rom.mesh.n_cells)
            self.log_lambdac_mean = torch.ones(self.data.n_supervised_samples, self.rom.mesh.n_cells,
                                                requires_grad=True)

            # Change for non unit square domains!!
            xx, yy = np.meshgrid(np.linspace(0, 1, self.data.output_resolution),
                                 np.linspace(0, 1, self.data.output_resolution))
            X = np.concatenate((np.expand_dims(xx.flatten(), axis=1), np.expand_dims(yy.flatten(), axis=1)), axis=1)
            X = torch.tensor(X)
            self.rom.mesh.get_interpolation_matrix(X)
            self.lambdacOpt = optim.Adam([self.log_lambdac_mean], lr=5e-3)
            self.taucf = torch.ones(self.data.output_resolution**2)

            if __debug__:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.writer = SummaryWriter('runs/gendrrom/' + current_time, flush_secs=5)        # for tensorboard

    def fit(self, n_steps=100):
        # method to train the model
        for s in range(n_steps):
            print('step = ', s)
            t = my.tic()
            # Only subsample unsupervised samples!
            batch_samples_z = torch.multinomial(torch.ones(self.data.n_unsupervised_samples), self.batch_size_z)\
                              + self.data.n_supervised_samples
            self.z_step(batch_samples_z)
            t = my.toc(t, 'z step')
            batch_samples_thetaf = torch.multinomial(torch.ones(self.data.n_unsupervised_samples),
                                                     self.batch_size_N_thetaf)
            self.thetaf_step(batch_samples_thetaf)
            t = my.toc(t, 'thetaf step')
            for k in range(100):
                batch_samples_lambdac = torch.multinomial(torch.ones(self.data.n_supervised_samples),
                                                          self.batch_size_N_lambdac)
                self.lambdac_step(batch_samples_lambdac)
            t = my.toc(t, 'lambdac step')
            print('uc = ', self.rom.full_solution.array)
            for k in range(100):
                batch_samples_thetac = torch.multinomial(torch.ones(self.data.n_supervised_samples),
                                                         self.batch_size_N_thetac)
                self.thetac_step(batch_samples_thetac)
            my.toc(t, 'thetac step')

    def predict(self, x):
        # method to predict from the model for a certain input x
        print('This is the prediction routine')

    def loss_thetaf(self, pred, batch_samples):
        eps = 1e-16
        return -torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(pred + eps), dim=1).flatten()) - \
                torch.dot(1 - self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(1 - pred + eps), dim=1).flatten())

    def loss_thetac(self, log_lambdac_pred, batch_samples):
        # needs to be updated to samples of lambda_c!!
        x = self.log_lambdac_mean[batch_samples, :]
        # this is for samples of z
        x = x.unsqueeze(1)
        x = x.expand(log_lambdac_pred.shape)
        diff = x - log_lambdac_pred
        return torch.dot((self.tauc * torch.mean(diff, dim=1)).flatten(),
                         torch.mean(diff, dim=1).flatten())

    def loss_lambdac(self, uf_pred, log_lambdac_pred, batch_samples):
        # this is joint loss of pc and pcf for lambda_c!

        diff_f = self.data.P[batch_samples, :] - uf_pred
        loss_lambdac = torch.dot((self.taucf * diff_f).flatten(), diff_f.flatten())
        diff_c = log_lambdac_pred - self.log_lambdac_mean
        loss_thetac = torch.dot((self.tauc * diff_c).flatten(), diff_c.flatten())
        return loss_lambdac + loss_thetac

    def loss_z(self, batch_samples):
        # ATTENTION: actually all samples should be chosen as batch size
        # negative latent log distribution over all z's
        eps = 1e-16
        pred_c = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])
        # precision of pc still needs to be added!!
        diff = self.log_lambdac_mean - pred_c
        out = .5 * torch.dot((self.tauc * diff).flatten(), diff.flatten())

        # for the supervised samples, take all
        lambdaf_pred = self.pfNet(self.z_mean[:self.data.n_supervised_samples, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(lambdaf_pred + eps).flatten()) +
                 torch.dot(1.0 - self.data.microstructure_image[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + eps).flatten())) \
               + .5*torch.sum(self.z_mean[:self.data.n_supervised_samples, :]**2)

        # for the unsupervised samples, take only batch size. Only sample unsupervised samples!!
        lambdaf_pred = self.pfNet(self.z_mean[batch_samples, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                           torch.log(lambdaf_pred + eps).flatten()) +
                 torch.dot(1.0 - self.data.microstructure_image[batch_samples, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + eps).flatten())) \
               + .5 * torch.sum(self.z_mean[batch_samples, :] ** 2)
        return out

    def thetaf_step(self, batch_samples):
        # One training step for pf
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)
        pred = self.pfNet(z)
        loss = self.loss_thetaf(pred, batch_samples)
        # print('loss_f = ', loss)

        if __debug__:
            # print('loss_thetaf = ', loss)
            self.writer.add_scalar('Loss/train_pf', loss)
            self.writer.close()

        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()

    def thetac_step(self, batch_samples):
        # One training step for pc
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)  # this is to store samples of z
        log_lambdac_pred = self.pcNet(z)
        loss = self.loss_thetac(log_lambdac_pred, batch_samples)
        # print('loss_c = ', loss)

        if __debug__:
            # print('loss_thetac = ', loss)
            self.writer.add_scalar('Loss/train_pc', loss)
            self.writer.close()

        loss.backward()
        self.pcOpt.step()
        with torch.no_grad():
            self.tauc = torch.tensor(self.batch_size_N_thetac, dtype=self.dtype) / \
                        torch.sum((log_lambdac_pred.squeeze() - self.log_lambdac_mean[batch_samples, :]) ** 2, dim=0)
        self.pcOpt.zero_grad()

    def lambdac_step(self, batch_samples):
        # One training step for pcf
        # Needs to be updated to samples of lambda_c from approximate posterior
        uf_pred = self.rom_autograd(torch.exp(self.log_lambdac_mean[batch_samples, :]))
        log_lambdac_pred = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])
        loss = self.loss_lambdac(uf_pred, log_lambdac_pred, batch_samples)
        # print('loss_lambda_c = ', loss)

        if __debug__:
            # print('loss_lambdac = ', loss)
            self.writer.add_scalar('Loss/train_pcf', loss)
            self.writer.close()

        loss.backward()
        self.lambdacOpt.step()
        with torch.no_grad():
            eps = 1e-16
            self.taucf = torch.tensor(self.batch_size_N_lambdac, dtype=self.dtype) / \
                                                torch.sum((uf_pred - self.data.P[batch_samples, :]) ** 2 + eps, dim=0)
        self.lambdacOpt.zero_grad()

    def z_step(self, batch_samples):
        # optimize latent distribution p(lambda_c^n, z^n) for point estimates

        loss_z = self.loss_z(batch_samples)
        loss_z.backward(retain_graph=True)
        # print('loss z = ', loss_z)

        self.zOpt.step()
        self.zOpt.zero_grad()

    def save(self, path='./model.p'):
        # save the whole model for later use, e.g. inference or training continuation
        state_dict = {'dtype': self.dtype,
                      'rom_state_dict': self.rom.state_dict(),
                      'dim_z': self.dim_z,
                      'z_mean': self.z_mean,
                      'zOpt_state_dict': self.zOpt.state_dict(),
                      'lr_z': self.lr_z,
                      'batch_size_z': self.batch_size_z,
                      'pfNet_state_dict': self.pfNet.state_dict(),
                      'pfNet_optimizer_state_dict': self.pfOpt.state_dict(),
                      'batch_size_N_thetaf': self.batch_size_N_thetaf,
                      'batch_size_N_lambdac': self.batch_size_N_lambdac,
                      'pcNet_state_dict': self.pcNet.state_dict(),
                      'pcNet_optimizer_state_dict': self.pcOpt.state_dict(),
                      'tauc': self.tauc,
                      'batch_size_N_thetac': self.batch_size_N_thetac,
                      'log_lambdac_mean': self.log_lambdac_mean,
                      'lambdacOpt_state_dict': self.lambdacOpt.state_dict(),
                      'taucf': self.taucf,
                      'writer': self.writer}

        torch.save(state_dict, path)

    def load(self, path='./model.p'):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        self.dtype = state_dict['dtype']
        self.rom = ROM.ROM()
        self.rom.load_state_dict(state_dict['rom_state_dict'])
        self.rom_autograd = self.rom.get_autograd_fun()
        self.data = dta.StokesData()
        self.dim_z = state_dict['dim_z']
        self.z_mean = state_dict['z_mean']
        self.lr_z = state_dict['lr_z']
        self.zOpt = optim.SGD([self.z_mean], self.lr_z)
        self.zOpt.load_state_dict(state_dict['zOpt_state_dict'])
        self.batch_size_z = state_dict['batch_size_z']
        self.pfNet = PfNet(self.dim_z, self.data.img_resolution**2)
        self.pfNet.load_state_dict(state_dict['pfNet_state_dict'])
        self.pfOpt = optim.Adam(self.pfNet.parameters())
        self.pfOpt.load_state_dict(state_dict['pfNet_optimizer_state_dict'])
        self.batch_size_N_thetaf = state_dict['batch_size_N_thetaf']
        self.batch_size_N_lambdac = state_dict['batch_size_N_lambdac']
        self.pcNet = PcNet(self.dim_z, self.rom.mesh.n_cells)
        self.pcNet.load_state_dict(state_dict['pcNet_state_dict'])
        self.pcOpt = optim.Adam(self.pcNet.parameters())
        self.pcOpt.load_state_dict(state_dict['pcNet_optimizer_state_dict'])
        self.tauc = state_dict['tauc']
        self.batch_size_N_thetac = state_dict['batch_size_N_thetac']
        self.log_lambdac_mean = state_dict['log_lambdac_mean']
        self.lambdacOpt = optim.Adam([self.log_lambdac_mean])
        self.lambdacOpt.load_state_dict(state_dict['lambdacOpt_state_dict'])
        self.taucf = state_dict['taucf']
        if __debug__:
            self.writer = state_dict['writer']

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














