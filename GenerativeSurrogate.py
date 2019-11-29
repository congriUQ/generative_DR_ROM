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
import pickle


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
        self.ac0 = nn.Softplus(beta=1e-1)

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
    def __init__(self, rom, data, dim_z):
        self.dtype = torch.float32

        self.rom = rom
        self.rom_autograd = rom.get_autograd_fun()
        self.data = data

        self.dim_z = dim_z
        self.z_mean = torch.zeros(self.data.n_supervised_samples + self.data.n_unsupervised_samples,
                                  dim_z, requires_grad=True)
        self.varDistOpt = optim.SGD([self.z_mean], lr=1e-3)
        self.batch_size_z = min(self.data.n_unsupervised_samples, 128)   # only point estimates for the time being

        self.pfNet = PfNet(dim_z, self.data.img_resolution**2)
        self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=4e-3)
        self.batch_size_N_thetaf = min(self.data.n_unsupervised_samples, 128)

        # so far no batched evaluation implemented. EXTEND THIS!!
        self.batch_size_N_lambdac = min(self.data.n_supervised_samples, 1)

        self.pcNet = PcNet(dim_z, rom.mesh.n_cells)
        self.pcOpt = optim.Adam(self.pcNet.parameters(), lr=1e-3)
        self.batch_size_N_thetac = min(self.data.n_supervised_samples, 256)
        self.log_lambda_c_mean = torch.ones(self.data.n_supervised_samples, self.rom.mesh.n_cells, requires_grad=True)

        # Change for non unit square domains!!
        xx, yy = np.meshgrid(np.linspace(0, 1, self.data.output_resolution),
                             np.linspace(0, 1, self.data.output_resolution))
        X = np.concatenate((np.expand_dims(xx.flatten(), axis=1), np.expand_dims(yy.flatten(), axis=1)), axis=1)
        X = torch.tensor(X)
        self.rom.mesh.get_interpolation_matrix(X)
        self.pcfOpt = optim.Adam([self.log_lambda_c_mean], lr=8e-2)

        if __debug__:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter('runs/gendrrom/' + current_time, flush_secs=5)        # for tensorboard

    def fit(self, n_steps=100):
        # method to train the model
        for s in range(n_steps):
            print('step = ', s)
            t = my.tic()
            batch_samples_z = torch.multinomial(torch.ones(self.data.n_unsupervised_samples), self.batch_size_z)
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
            for k in range(100):
                batch_samples_thetac = torch.multinomial(torch.ones(self.data.n_supervised_samples),
                                                         self.batch_size_N_thetac)
                self.thetac_step(batch_samples_thetac)
            my.toc(t, 'thetac step')

    def predict(self, x):
        # method to predict from the model for a certain input x
        pass

    def loss_thetaf(self, pred, batch_samples):
        eps = 1e-16
        return -torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(pred + eps), dim=1).flatten()) - \
                torch.dot(1 - self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(1 - pred + eps), dim=1).flatten())

    def loss_thetac(self, pred, batch_samples):
        # needs to be updated to samples of lambda_c!!
        x = self.log_lambda_c_mean[batch_samples, :]
        x = x.unsqueeze(1)
        x = x.expand(pred.shape)
        return torch.dot(torch.mean(x - pred, dim=1).flatten(),
                         torch.mean(x - pred, dim=1).flatten())

    def loss_lambdac(self, uf_pred, pred_c, batch_samples):
        # this is joint loss of pc and pcf for lambda_c!

        loss_lambdac = torch.dot(self.data.P[batch_samples, :].flatten() - uf_pred,
                             self.data.P[batch_samples, :].flatten() - uf_pred)
        x = self.log_lambda_c_mean.flatten()
        loss_thetac = torch.dot(pred_c.flatten() - x, pred_c.flatten() - x)
        return loss_lambdac + loss_thetac

    def loss_z(self, Z, batch_samples):
        # negative latent log distribution over all z's
        eps = 1e-16
        pred_c = self.pcNet(Z[:self.data.n_supervised_samples, :])
        # precision of pc still needs to be added!!
        out = .5*torch.dot((self.log_lambda_c_mean - pred_c).flatten(), (self.log_lambda_c_mean - pred_c).flatten())
        pred_f = self.pfNet(Z)
        # out -= ... !!
        out -= (torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.log(pred_f + eps).flatten()) +
                torch.dot(1 - self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.log(1 - pred_f + eps).flatten())) - .5*torch.sum(Z*Z)
        return out

    def thetaf_step(self, batch_samples):
        # One training step for pf
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)
        pred = self.pfNet(z)
        loss = self.loss_thetaf(pred, batch_samples)

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
        z = z.unsqueeze(1)
        pred = self.pcNet(z)
        loss = self.loss_thetac(pred, batch_samples)

        if __debug__:
            # print('loss_thetac = ', loss)
            self.writer.add_scalar('Loss/train_pc', loss)
            self.writer.close()

        loss.backward()
        self.pcOpt.step()
        self.pcOpt.zero_grad()

    def lambdac_step(self, batch_samples):
        # One training step for pcf
        # Needs to be updated to samples of lambda_c from approximate posterior
        pred_cf = self.rom_autograd(torch.exp(self.log_lambda_c_mean[batch_samples, :]))
        pred_c = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])
        loss = self.loss_lambdac(pred_cf, pred_c, batch_samples)

        if __debug__:
            # print('loss_lambdac = ', loss)
            self.writer.add_scalar('Loss/train_pcf', loss)
            self.writer.close()

        loss.backward()
        self.pcfOpt.step()
        self.pcfOpt.zero_grad()

    def z_step(self, batch_samples):
        # optimize latent distribution p(lambda_c^n, z^n) for point estimates

        loss_z = self.loss_z(self.z_mean[batch_samples, :], batch_samples)
        loss_z.backward(retain_graph=True)
        self.varDistOpt.step()
        self.varDistOpt.zero_grad()

    def save(self):
        # save the whole model for later use, e.g. inference or training continuation
        state_dict = {'pfNet_state_dict': self.pfNet.state_dict(),
                      'pcNet_state_dict': self.pcNet.state_dict(),
                      'pfNet_optimizer_state_dict': self.pfOpt.state_dict(),
                      'pcNet_optimizer_state_dict': self.pcOpt.state_dict(),
                      'pcfOpt_state_dict': self.pcfOpt.state_dict(),
                      'varDistOpt_state_dict': self.varDistOpt.state_dict(),
                      'z_mean': self.z_mean,
                      'log_lambda_c_mean': self.log_lambda_c_mean,
                      'writer': self.writer}

        torch.save(state_dict, './model')


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














