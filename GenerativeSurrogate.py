'''All generative dimension reduction ROM surrogate model components'''
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import myutil as my
import ROM
import Data as dta
import warnings
import unittest


# noinspection PyUnreachableCode
class GenerativeSurrogate:
    def __init__(self, rom=None, data=None, dim_z=20):
        # rom is only allowed to be None if the model is loaded afterwards
        if rom is not None:
            self.dtype = torch.float32

            self.rom = rom
            self.rom_autograd = rom.get_autograd_fun() if rom is not None else None

            if data is None:
                self.data = dta.StokesData()
            else:
                self.data = data

            if self.data.n_supervised_samples == 0:
                warnings.warn('No supervised training data. Train autoencoder only')

            self.dim_z = dim_z
            self.z_mean = torch.randn(self.data.n_supervised_samples + self.data.n_unsupervised_samples,
                                      self.dim_z, requires_grad=True)
            self.z_mean.data = 1e-2 * self.z_mean.data
            self.lr_z = 1e-1
            # self.zOpt = optim.Adam([self.z_mean], lr=self.lr_z, betas=(.3, .5))
            self.zOpt = optim.SGD([self.z_mean], lr=self.lr_z)
            self.zOptSched = optim.lr_scheduler.ReduceLROnPlateau(self.zOpt, factor=.2, verbose=True)
            self.batch_size_z = min(self.data.n_unsupervised_samples, 128)

            self.pfNet = PfNet(self.dim_z, self.data.img_resolution**2)
            self.pfOpt = optim.Adam(self.pfNet.parameters(), lr=3e-3)
            self.pfOptSched = optim.lr_scheduler.ReduceLROnPlateau(self.pfOpt, factor=.5, verbose=True, patience=30)
            self.batch_size_N_thetaf = min(self.data.n_supervised_samples + self.data.n_unsupervised_samples, 512)

            # so far no batched evaluation implemented. EXTEND THIS!!
            self.batch_size_N_lambdac = min(self.data.n_supervised_samples, 1)
            self.batch_size_N_thetac = min(self.data.n_supervised_samples, 128)

            self.pcNet = PcNet(self.dim_z, rom.mesh.n_cells)
            self.pcOpt = optim.Adam(self.pcNet.parameters(), lr=1e-3)
            self.thetacSched = optim.lr_scheduler.ReduceLROnPlateau(self.pcOpt, factor=.2, patience=1000, verbose=True)
            self.tauc = torch.ones(self.rom.mesh.n_cells)

            # self.log_lambdac_mean = torch.ones(self.data.n_supervised_samples, self.rom.mesh.n_cells,
            #                                     requires_grad=True)
            # self.log_lambdac_mean.data = -10.0 * self.log_lambdac_mean.data
            self.log_lambdac_mean = []
            self.log_lambdac_mean_tensor = torch.empty(self.data.n_supervised_samples, self.rom.mesh.n_cells)
            for n in range(self.data.n_supervised_samples):
                log_lambdac_init = torch.ones(self.rom.mesh.n_cells, requires_grad=True)
                log_lambdac_init.data = -10.0 * log_lambdac_init.data
                log_lambdac_pred = self.pcNet(self.z_mean[n, :])
                self.log_lambdac_mean.append(LogLambdacParam(self.data.P[n, :], log_lambdac_pred,
                                                             init_value=log_lambdac_init))
                self.log_lambdac_mean[-1].optimizer = optim.Adam([self.log_lambdac_mean[-1].value], lr=1e-2)
                self.log_lambdac_mean[-1].scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.log_lambdac_mean[-1].optimizer, factor=.1, verbose=True, min_lr=1e-12, patience=15)
                self.log_lambdac_mean[-1].finescale_output = self.data.P[n, :]
                # for z updates
                self.log_lambdac_mean_tensor[n, :] = self.log_lambdac_mean[-1].value

            # Change for non unit square domains!!
            xx, yy = np.meshgrid(np.linspace(0, 1, self.data.output_resolution),
                                 np.linspace(0, 1, self.data.output_resolution))
            X = np.concatenate((np.expand_dims(xx.flatten(), axis=1), np.expand_dims(yy.flatten(), axis=1)), axis=1)
            X = torch.tensor(X)
            self.rom.mesh.get_interpolation_matrix(X)
            # self.lambdacOpt = optim.Adam(self.log_lambdac_mean.split, lr=1e-2)
            # self.lambdacOptSched = optim.lr_scheduler.ReduceLROnPlateau(self.lambdacOpt, factor=.1, verbose=True,
            #                                                             min_lr=1e-9, patience=15)
            self.taucf = torch.ones(self.data.output_resolution**2)

            self.training_iterations = 0

            self.writeParams = False
            if __debug__ and self.writeParams:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.writer = SummaryWriter('runs/gendrrom/' + current_time, flush_secs=5)        # for tensorboard
            else:
                self.writer = None

    def is_equal(self, model):
        """
        Check if model is identical and in same state as the model 'model', e.g., a loaded model.
        There is no 100% certainty because builtin functions cannot be compared.
        TO BE COMPLETED
        """
        for attr in self.__dict__:
            print('attr == ', attr)
            if type(self.__dict__[attr]) == torch.Tensor:
                #
                if not torch.all(self.__dict__[attr] == model.__dict__[attr]):
                    print(attr, ' not equal')

            elif self.__dict__[attr] != model.__dict__[attr]:
                print(attr, ' not equal')

    def fit(self, n_steps=100, z_iterations=10, thetaf_iterations=100, lambdac_iterations=100, thetac_iterations=100,
            save_iterations=20, with_precisions=True):
        """
        method to train the model with_precisions updates the precisions taucf and tauc
        """
        for s in range(n_steps):
            print('step = ', self.training_iterations)
            t = my.tic()
            # Only subsample unsupervised samples!
            if self.training_iterations < 3:
                lr_init = 1e-3
            else:
                lr_init = 1e-4
            self.zOpt.param_groups[0]['lr'] = lr_init
            z_iter = 0
            while z_iter < z_iterations and self.zOpt.param_groups[0]['lr'] > 5e-3 * lr_init:
                # batch_samples_z = torch.multinomial(torch.ones(self.data.n_unsupervised_samples), self.batch_size_z)\
                #                   + self.data.n_supervised_samples
                batch_samples_z = None      # we currently use all z samples
                self.z_step(batch_samples_z, z_iter)
                z_iter += 1
            t = my.toc(t, 'z step')

            thetaf_iter = 0
            if self.training_iterations < 5:
                lr_init = 5e-3
            elif self.training_iterations < 10:
                lr_init = 3e-3
            else:
                lr_init = 5e-4
            self.pfOpt.param_groups[0]['lr'] = lr_init
            while thetaf_iter < thetaf_iterations and self.pfOpt.param_groups[0]['lr'] > 1e-2 * lr_init:
                batch_samples_thetaf = torch.multinomial(torch.ones(
                    self.data.n_supervised_samples + self.data.n_unsupervised_samples), self.batch_size_N_thetaf)
                self.thetaf_step(batch_samples_thetaf, thetaf_iter)
                thetaf_iter += 1
            t = my.toc(t, 'thetaf step')

            if self.data.n_supervised_samples:
                # Don't train supervised part if there is no supervised data. Instead, train autoencoder only.
                # for k in range(self.data.n_supervised_samples):
                #     # batch_samples_lambdac = torch.multinomial(torch.ones(self.data.n_supervised_samples),
                #     #                                           self.batch_size_N_lambdac)
                #     print('sample == ', k)
                #     batch_samples_lambdac = k
                #     if s < 3:
                #         lr_init = 1e-2
                #     else:
                #         lr_init = 1e-4
                #     self.lambdacOpt.param_groups[0]['lr'] = lr_init
                #     lambdac_iter = 0
                #     while lambdac_iter < lambdac_iterations and self.lambdacOpt.param_groups[0]['lr'] > 1e-4 * lr_init:
                #         self.lambdac_step(batch_samples_lambdac, lambdac_iter)
                #         lambdac_iter += 1
                for n in range(self.data.n_supervised_samples):
                    print('sample == ', n)
                    self.log_lambdac_mean[n].log_lambdac_pred = self.pcNet(self.z_mean[n, :])
                    self.log_lambdac_mean[n].converge(self, self.data.n_supervised_samples, mode=n)
                    self.log_lambdac_mean_tensor[n, :] = self.log_lambdac_mean[n].value

                t = my.toc(t, 'lambdac step')

                if with_precisions:
                    with torch.no_grad():
                        eps = 1e-10
                        uf_pred = torch.zeros_like(self.data.P)
                        for n in range(self.data.n_supervised_samples):
                            uf_pred[n] = self.rom_autograd(torch.exp(self.log_lambdac_mean[n].value))
                        self.taucf = torch.tensor(self.data.n_supervised_samples, dtype=self.dtype) / \
                                                            torch.sum((uf_pred - self.data.P) ** 2 + eps, dim=0)

                thetac_iter = 0
                if self.training_iterations < 3:
                    lr_init = 7e-4
                elif self.training_iterations < 5:
                    lr_init = 5e-4
                else:
                    lr_init = 1e-4
                self.pcOpt.param_groups[0]['lr'] = lr_init
                while thetac_iter < thetac_iterations and self.pcOpt.param_groups[0]['lr'] > 2e-4 * lr_init:
                    batch_samples_thetac = torch.multinomial(torch.ones(self.data.n_supervised_samples),
                                                             self.batch_size_N_thetac)
                    self.thetac_step(batch_samples_thetac, thetac_iter)
                    thetac_iter += 1
                my.toc(t, 'thetac step')

                if with_precisions:
                    with torch.no_grad():
                        log_lambdac_pred = self.pcNet(self.z_mean[:self.data.n_supervised_samples])
                        self.tauc = torch.tensor(self.batch_size_N_thetac, dtype=self.dtype) / \
                                    torch.sum((log_lambdac_pred.squeeze() - self.log_lambdac_mean_tensor) ** 2, dim=0)

            if (self.training_iterations + 1) % save_iterations == 0:
                # + 1 to not save in the very first iteration
                print('Saving model...')
                self.save()
                print('...saving done.')
            self.training_iterations += 1

    def predict(self, testData, max_iterations=1000, optimizer='Adam', lr=1e-1, Z_init=None):
        # method to predict from the model for a certain set of input samples packaged in a StokesData object testData
        # Z holds the latent z representations of the inputs lambdaf

        # Initialize Z
        Z = torch.randn(testData.n_unsupervised_samples, self.dim_z, requires_grad=True)
        if Z_init is not None:
            Z.data = Z_init.data
        # Ztmp = torch.mean(self.z_mean, dim=0)
        # Ztmp = Ztmp.unsqueeze(0)
        # Z.data = Ztmp.repeat((testData.n_unsupervised_samples, 1)).data

        # Take same learning rate as was used for training
        if optimizer == 'Adam':
            zOptPredict = optim.Adam([Z], lr=lr, betas=(.7, .9))
        elif optimizer == 'SGD':
            zOptPredict = optim.SGD([Z], lr=lr)
        else:
            raise Exception

        lr_init = zOptPredict.param_groups[0]['lr']
        zOptSched = optim.lr_scheduler.ReduceLROnPlateau(zOptPredict, factor=.2, patience=300, verbose=True)

        grad_norm = []
        step = 0
        while step < max_iterations and zOptPredict.param_groups[0]['lr'] > 1e-6 * lr_init:
            self.z_prediction_step(Z, testData, zOptPredict, zOptSched, step=step)
            if __debug__:
                grad_norm.append(torch.norm(Z.grad))
            step += 1

        log_lambdac_mean = self.pcNet(Z)
        # Add noise here...

        uf_pred = []
        for log_lambdac_mean_sample in log_lambdac_mean:
            uf_pred.append(self.rom_autograd(torch.exp(log_lambdac_mean_sample)).detach().numpy())

        return uf_pred, Z, grad_norm

    def loss_thetaf(self, lambdaf_pred, batch_samples):
        eps = 1e-16
        return -torch.dot(self.data.microstructure_image[batch_samples, :].flatten(),
                          torch.mean(torch.log(lambdaf_pred + eps), dim=1).flatten()) \
               -torch.dot(self.data.microstructure_image_inverted[batch_samples, :].flatten(),
                          torch.mean(torch.log(1.0 - lambdaf_pred + eps), dim=1).flatten())

    def loss_thetac(self, log_lambdac_pred, batch_samples):
        # needs to be updated to samples of lambda_c!!
        x = self.log_lambdac_mean_tensor[batch_samples, :]
        # this is for samples of z
        x = x.unsqueeze(1)
        x = x.expand(log_lambdac_pred.shape)
        diff = x - log_lambdac_pred
        return torch.dot((self.tauc * torch.mean(diff, dim=1)).flatten(),
                         torch.mean(diff, dim=1).flatten())

    def loss_lambdac(self, uf_pred, log_lambdac_pred, batch_samples):
        # this is joint loss of pc and pcf for lambda_c!
        # depreceated

        diff_f = self.data.P[batch_samples, :] - uf_pred
        loss_lambdac = torch.dot((self.taucf * diff_f).flatten(), diff_f.flatten())
        diff_c = log_lambdac_pred - self.log_lambdac_mean
        loss_thetac = torch.dot((self.tauc * diff_c).flatten(), diff_c.flatten())
        return loss_lambdac + loss_thetac

    def loss_z_delta(self, batch_samples):
        # Loss w.r.t. z assuming q(z) = delta(z)
        # ATTENTION: actually all samples should be chosen as batch size
        # negative latent log distribution over all z's
        eps = 1e-16
        pred_lambda_c = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])

        diff = self.log_lambdac_mean_tensor - pred_lambda_c
        out = .5 * torch.dot((self.tauc * diff).flatten(), diff.flatten())

        # for the supervised samples, take all
        lambdaf_pred = self.pfNet(self.z_mean[:self.data.n_supervised_samples, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(lambdaf_pred + eps).flatten()) +
                 torch.dot(self.data.microstructure_image_inverted[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + eps).flatten())) \
               + .5*torch.sum(self.z_mean[:self.data.n_supervised_samples, :]**2)

        # for the unsupervised samples, take only batch size. Only sample unsupervised samples!!
        lambdaf_pred = self.pfNet(self.z_mean[self.data.n_supervised_samples:, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[self.data.n_supervised_samples:, :].flatten(),
                           torch.log(lambdaf_pred + eps).flatten()) +
                 torch.dot(self.data.microstructure_image_inverted[self.data.n_supervised_samples:, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + eps).flatten())) \
                            + .5 * torch.sum(self.z_mean[self.data.n_supervised_samples:, :] ** 2)
        return out

    def log_q_z_emp(self, z):
        # TO BE DONE
        pred_lambda_c = self.pcNet(z)
        diff = self.log_lambdac_mean_tensor - pred_lambda_c
        out = .5 * torch.dot((self.tauc * diff).flatten(), diff.flatten())

    def loss_z_prediction(self, Z, testData):
        # negative latent log distribution over all z's
        eps = 1e-16

        # for the supervised samples, take all
        lambdaf_pred = self.pfNet(Z)
        # out += - ... for readability
        out = -(torch.dot(testData.microstructure_image.flatten(), torch.log(lambdaf_pred + eps).flatten()) +
                torch.dot(testData.microstructure_image_inverted.flatten(),
                          torch.log(1.0 - lambdaf_pred + eps).flatten())) + .5*torch.sum(Z**2)
        return out

    def thetaf_step(self, batch_samples, step=1):
        # One training step for pf
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)
        lambdaf_pred = self.pfNet(z)
        loss = self.loss_thetaf(lambdaf_pred, batch_samples)
        assert torch.isfinite(loss)
        if step % 5 == 0:
            print('loss_f = ', loss.item())

        if __debug__ and self.writeParams:
            # print('loss_thetaf = ', loss)
            self.writer.add_scalar('Loss/train_thetaf', loss)
            self.writer.close()

        loss.backward()
        self.pfOpt.step()
        self.pfOpt.zero_grad()
        self.pfOptSched.step(loss)

    def thetac_step(self, batch_samples, step=1):
        # One training step for pc
        # batch_samples are indices of the samples contained in the batch
        # This needs to be replaced by the (approximate) posterior on z!!
        z = self.z_mean[batch_samples, :]
        z = z.unsqueeze(1)  # this is to store samples of z
        log_lambdac_pred = self.pcNet(z)
        loss = self.loss_thetac(log_lambdac_pred, batch_samples)
        assert torch.isfinite(loss)
        if step % 50 == 0:
            print('loss_c = ', loss.item())

        if __debug__ and self.writeParams:
            # print('loss_thetac = ', loss)
            self.writer.add_scalar('Loss/train_pc', loss)
            self.writer.close()

        loss.backward(retain_graph=True)
        self.pcOpt.step()
        self.pcOpt.zero_grad()
        self.thetacSched.step(loss)

    def lambdac_step(self, batch_samples, step=1):
        # One training step for pcf
        # Needs to be updated to samples of lambda_c from approximate posterior
        # depreceated
        eps = 1e-12     # for solver stability
        self.lambdacOpt.zero_grad()
        uf_pred = self.rom_autograd(torch.exp(self.log_lambdac_mean[batch_samples, :]) + eps)
        assert torch.all(torch.isfinite(uf_pred))
        log_lambdac_pred = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])
        assert torch.all(torch.isfinite(uf_pred))
        loss = self.loss_lambdac(uf_pred, log_lambdac_pred, batch_samples)
        assert torch.isfinite(loss)
        loss.backward()
        if (step % 400) == 0:
            print('loss_lambda_c = ', loss.item())

        # if __debug__:
        #     # print('loss_lambdac = ', loss)
        #     self.writer.add_scalar('Loss/train_pcf', loss)
        #     self.writer.close()

        self.lambdacOpt.step()
        self.lambdacOptSched.step(loss)

    def z_step(self, batch_samples, step=1):
        # optimize latent distribution p(lambda_c^n, z^n) for point estimates

        loss_z = self.loss_z_delta(batch_samples)
        assert torch.isfinite(loss_z)
        loss_z.backward(retain_graph=True)
        if step % 20:
            print('loss z = ', loss_z.item())

        self.zOpt.step()
        self.zOpt.zero_grad()
        self.zOptSched.step(loss_z)

    def z_prediction_step(self, Z, testData, zOptPredict, zOptSched, step=1):
        zOptPredict.zero_grad()
        loss_z = self.loss_z_prediction(Z, testData)
        assert torch.isfinite(loss_z)
        loss_z.backward()
        if step % 100 == 0:
            print('loss_z_prediction = ', loss_z.item())
            # print('norm grad_z prediction = ', torch.norm(Z.grad))

        zOptPredict.step()
        zOptSched.step(loss_z)

    def save(self, path='./model.p'):
        # save the whole model for later use, e.g. inference or training continuation
        state_dict = {'dtype': self.dtype,
                      'rom_state_dict': self.rom.state_dict(),
                      'supervised_samples': self.data.supervised_samples,
                      'unsupervised_samples': self.data.unsupervised_samples,
                      'dim_z': self.dim_z,
                      'z_mean': self.z_mean,
                      'zOpt_state_dict': self.zOpt.state_dict(),
                      'zOptSched_state_dict': self.zOptSched.state_dict(),
                      'lr_z': self.lr_z,
                      'batch_size_z': self.batch_size_z,
                      'pfNet_state_dict': self.pfNet.state_dict(),
                      'pfNet_optimizer_state_dict': self.pfOpt.state_dict(),
                      'pfOptSched_state_dict': self.pfOptSched.state_dict(),
                      'batch_size_N_thetaf': self.batch_size_N_thetaf,
                      'batch_size_N_lambdac': self.batch_size_N_lambdac,
                      'pcNet_state_dict': self.pcNet.state_dict(),
                      'pcNet_optimizer_state_dict': self.pcOpt.state_dict(),
                      'thetacSched_state_dict': self.thetacSched.state_dict(),
                      'tauc': self.tauc,
                      'batch_size_N_thetac': self.batch_size_N_thetac,
                      'log_lambdac_mean': self.log_lambdac_mean,
                      'log_lambdac_mean_tensor': self.log_lambdac_mean_tensor,
                      'taucf': self.taucf,
                      'training_iterations': self.training_iterations,
                      # 'writer': self.writer
                      }

        torch.save(state_dict, path)

    def load(self, path='./model.p', mode='train'):
        # mode can be 'train' to continue training or 'predict' to do inference with the model
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, mode)

    def load_state_dict(self, state_dict, mode='train'):
        # mode can be 'train' to continue training or 'predict' to do inference with the model
        self.dtype = state_dict['dtype']
        self.rom = ROM.ROM()
        self.rom.load_state_dict(state_dict['rom_state_dict'])
        self.rom_autograd = self.rom.get_autograd_fun()
        if mode == 'train':
            self.data = dta.StokesData(state_dict['supervised_samples'], state_dict['unsupervised_samples'])
            self.data.read_data()
            self.data.reshape_microstructure_image()
        self.dim_z = state_dict['dim_z']
        self.z_mean = state_dict['z_mean']
        self.lr_z = state_dict['lr_z']
        self.zOpt = optim.Adam([self.z_mean], lr=self.lr_z, betas=(.3, .5))
        self.zOpt.load_state_dict(state_dict['zOpt_state_dict'])
        self.zOptSched = optim.lr_scheduler.ReduceLROnPlateau(self.zOpt, factor=.2, verbose=True)
        self.zOptSched.load_state_dict(state_dict['zOptSched_state_dict'])
        self.batch_size_z = state_dict['batch_size_z']
        self.pfNet = PfNet(self.dim_z, self.data.img_resolution**2)
        self.pfNet.load_state_dict(state_dict['pfNet_state_dict'])
        self.pfOpt = optim.Adam(self.pfNet.parameters())
        self.pfOpt.load_state_dict(state_dict['pfNet_optimizer_state_dict'])
        self.pfOptSched = optim.lr_scheduler.ReduceLROnPlateau(self.zOpt, factor=.2, verbose=True, min_lr=1e-9)
        self.pfOptSched.load_state_dict(state_dict['pfOptSched_state_dict'])
        self.batch_size_N_thetaf = state_dict['batch_size_N_thetaf']
        self.batch_size_N_lambdac = state_dict['batch_size_N_lambdac']
        self.pcNet = PcNet(self.dim_z, self.rom.mesh.n_cells)
        self.pcNet.load_state_dict(state_dict['pcNet_state_dict'])
        self.pcOpt = optim.Adam(self.pcNet.parameters())
        self.pcOpt.load_state_dict(state_dict['pcNet_optimizer_state_dict'])
        self.thetacSched = optim.lr_scheduler.ReduceLROnPlateau(self.pcOpt, factor=.2, verbose=True, min_lr=1e-9)
        self.thetacSched.load_state_dict(state_dict['thetacSched_state_dict'])
        self.tauc = state_dict['tauc']
        self.batch_size_N_thetac = state_dict['batch_size_N_thetac']
        self.log_lambdac_mean = state_dict['log_lambdac_mean']
        self.log_lambdac_mean_tensor = state_dict['log_lambdac_mean_tensor']
        self.taucf = state_dict['taucf']
        self.training_iterations = state_dict['training_iterations']
        if __debug__ and self.writeParams:
            try:
                self.writer = state_dict['writer']
            except KeyError:
                # If no writer was saved, set writer to None
                self.writer = None

    def plot_input_reconstruction(self):
        fig, ax = plt.subplots(2, 4)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # This needs to be replaced by the (approximate) posterior on z!!
        reconstructions = self.pfNet(self.z_mean[:4])

        for col in range(4):
            # reconstructions
            im0 = ax[0][col].imshow(torch.reshape(reconstructions[col],
                                                  (self.data.img_resolution,
                                                   self.data.img_resolution)).detach().numpy(),
                                    cmap='gray_r', vmin=0, vmax=1)
            ax[0][col].set_xticks([], [])
            ax[0][col].set_yticks([], [])
            ax[0][col].set_title('reconstruction sample ' + str(col))

            # training data
            im1 = ax[1][col].imshow(torch.reshape(self.data.microstructure_image[col],
                                                  (self.data.img_resolution,
                                                   self.data.img_resolution)).detach().numpy(),
                                    cmap='gray_r', vmin=0, vmax=1)
            ax[1][col].set_xticks([], [])
            ax[1][col].set_yticks([], [])
            ax[1][col].set_title('traning sample ' + str(col))
        cbar_ax = fig.add_axes([0.92, .108, 0.01, 0.77])
        fig.colorbar(im0, cax=cbar_ax)
        plt.show()

    def plot_generated_microstructures(self):
        """
        Generates microstructures via z ~ p(z), lambdaf ~ pfNet(z)
        """
        z = torch.randn(size=(4, self.dim_z))
        lambdaf_samples = self.pfNet(z)

        fig, ax = plt.subplots(1, 4)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        for col in range(4):
            im = ax[col].imshow(torch.reshape(lambdaf_samples[col],
                                              (self.data.img_resolution,
                                               self.data.img_resolution)).detach().numpy(),
                                cmap='gray_r', vmin=0, vmax=1)
            ax[col].set_xticks([], [])
            ax[col].set_yticks([], [])
            ax[col].set_title('generated sample ' + str(col))

        cbar_ax = fig.add_axes([0.92, .325, 0.01, 0.34])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()


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
        self.ac0 = nn.Sigmoid()
        self.fc1 = nn.Linear(rom_n_cells, rom_n_cells)
        # self.ac1 = nn.Sigmoid()
        # self.fc2 = nn.Linear(rom_n_cells, rom_n_cells)
        # self.fc0 = nn.Linear(dim_z, rom_n_cells)

    def forward(self, z):
        x = self.fc0(z)
        x = self.ac0(x)
        x = self.fc1(x)
        # x = self.ac1(x)
        # x = self.fc2(x)
        return x


class PfNet(nn.Module):
    # From latent z-space to fine scale input data lambda_f
    def __init__(self, dim_z, dim_img):
        # dim_img = img_resolution**2, i.e., total number of pixels
        super(PfNet, self).__init__()
        self.dim_img = dim_img
        # dim_h = int(torch.sqrt(torch.tensor(dim_z*dim_img, dtype=torch.float32)))       # geometric mean
        dim_h = 2 * dim_z
        self.fc0 = nn.Linear(dim_z, dim_h)
        self.ac0 = nn.ReLU()
        self.fc1 = nn.Linear(dim_h, dim_img)
        self.ac1 = nn.Sigmoid()

        # zero initialization
        # self.fc0.weight.data = .0 * self.fc0.weight.data
        # self.fc0.bias.data = .0 * self.fc0.bias.data
        # self.fc1.weight.data = .0 * self.fc1.weight.data
        # self.fc1.bias.data = .0 * self.fc1.bias.data

    def forward(self, z):
        out = self.fc0(z)           # z.shape = (batch_size_N, batchSizeZ, dim_z)
        out = self.ac0(out)
        out = self.fc1(out)
        out = self.ac1(out)         # out.shape = (batch_size_N, batchSizeZ, img_resolution**2)
        return out


class ModelParam:
    """
    Class for a certain parameter of the model, i.e., thetaf, thetac, log_lambdac_mean, and z_mean
    """
    def __init__(self, init_value=None, batch_size=128):
        self.value = init_value                                 # Torch tensor of parameter values
        self.lr_init = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]     # Initial learning rate in every convergence iteration
        self.optimizer = None                                   # needs to be specified later
        self.scheduler = None
        self.batch_size = batch_size                            # the batch size for optimization of the parameter
        self.convergence_iteration = 0                          # iterations with full convergence
        self.step_iteration = 0                                 # iterations in the current optimization
        self.max_iter = 100
        self.lr_drop_factor = 3e-4                              # lr can drop by this factor until convergence
        self.eps = 1e-12                                        # for stability of log etc.

    def loss(self, **kwargs):
        """
        The loss function pertaining to the parameter
        Override this!
        """
        raise Exception('loss function not implemented')

    def step(self, **kwargs):
        """
        Performs a single optimization iteration
        Override this!
        """
        raise Exception('step function not implemented')

    def draw_batch_samples(self, n_samples_tot, mode='shuffle'):
        """
        Draws a batch sample. n_samples_tot is the size of the data set
        """
        if mode == 'shuffle':
            while True:
                yield torch.multinomial(torch.ones(n_samples_tot), self.batch_size)
        elif mode == 'iter':
            k = 0
            while True:
                k += 1
                yield torch.tensor(range(k*self.batch_size, (k + 1)*self.batch_size)) % n_samples_tot
        elif isinstance(mode, int):
            while True:
                yield mode

    def converge(self, model, n_samples_tot, mode='shuffle'):
        """
        Runs optimization of parameter till a convergence criterion is met
        """
        self.optimizer.param_groups[0]['lr'] = self.lr_init[min(self.convergence_iteration, len(self.lr_init) - 1)]
        batch_gen = self.draw_batch_samples(n_samples_tot, mode)
        while self.step_iteration < self.max_iter and self.optimizer.param_groups[0]['lr'] > \
                self.lr_drop_factor * self.lr_init[min(self.convergence_iteration, len(self.lr_init) - 1)]:
            batch_samples = next(batch_gen)
            self.step(model, batch_samples)
            self.step_iteration += 1
        self.convergence_iteration += 1
        self.step_iteration = 0


class LogLambdacParam(ModelParam):
    """
    Class pertaining to the log_lambdac_mean parameters
    """

    def __init__(self, finescale_output, log_lambdac_pred, init_value=None):
        super().__init__(init_value=init_value, batch_size=1)
        self.finescale_output = finescale_output
        self.log_lambdac_pred = log_lambdac_pred        # needs to be set before every convergence iteration

    def loss(self, model, uf_pred):
        """
        This is joint loss of pc and pcf for lambda_c!
        model:  GenerativeSurrogate object
        """
        diff_f = self.finescale_output - uf_pred
        loss_lambdac = torch.dot((model.taucf * diff_f).flatten(), diff_f.flatten())
        diff_c = self.log_lambdac_pred - self.value
        loss_thetac = torch.dot((model.tauc * diff_c).flatten(), diff_c.flatten())
        return loss_lambdac + loss_thetac

    def step(self, model, batch_samples):
        """
        One training step for log_lambdac_mean
        Needs to be updated to samples of lambda_c from approximate posterior
        model:  GenerativeSurrogate object
        """
        self.optimizer.zero_grad()
        uf_pred = model.rom_autograd(torch.exp(self.value) + self.eps)
        assert torch.all(torch.isfinite(uf_pred))
        loss = self.loss(model, uf_pred)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
        if (self.step_iteration % 500) == 0:
            print('loss_lambda_c = ', loss.item())

        # if __debug__:
        #     # print('loss_lambdac = ', loss)
        #     self.writer.add_scalar('Loss/train_pcf', loss)
        #     self.writer.close()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(loss)


class ZParam(ModelParam):
    def __init__(self, init_value=None):
        # batch size should be all for the time being
        super().__init__(init_value=init_value, batch_size=None)

    def loss(self):
        # ATTENTION: actually all samples should be chosen as batch size
        # negative latent log distribution over all z's
        pred_c = self.pcNet(self.z_mean[:self.data.n_supervised_samples, :])
        # precision of pc still needs to be added!!
        diff = self.log_lambdac_mean_tensor - pred_c
        out = .5 * torch.dot((self.tauc * diff).flatten(), diff.flatten())

        # for the supervised samples, take all
        lambdaf_pred = self.pfNet(self.z_mean[:self.data.n_supervised_samples, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(lambdaf_pred + self.eps).flatten()) +
                 torch.dot(self.data.microstructure_image_inverted[:self.data.n_supervised_samples, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + self.eps).flatten())) \
               + .5 * torch.sum(self.z_mean[:self.data.n_supervised_samples, :] ** 2)

        # for the unsupervised samples, take only batch size. Only sample unsupervised samples!!
        lambdaf_pred = self.pfNet(self.z_mean[self.data.n_supervised_samples:, :])
        # out += - ... for readability
        out += -(torch.dot(self.data.microstructure_image[self.data.n_supervised_samples:, :].flatten(),
                           torch.log(lambdaf_pred + self.eps).flatten()) +
                 torch.dot(self.data.microstructure_image_inverted[self.data.n_supervised_samples:, :].flatten(),
                           torch.log(1.0 - lambdaf_pred + self.eps).flatten())) \
               + .5 * torch.sum(self.z_mean[self.data.n_supervised_samples:, :] ** 2)
        return out


#############################################################################################
# UNIT TESTS
#############################################################################################
from poisson_fem import PoissonFEM


class ModelTestCase(unittest.TestCase):

    def setUp(self):
        # Some parameters
        lin_dim_rom = 2  # Linear number of rom elements
        a = np.array([1, 1, 0])  # Boundary condition function coefficients
        dtype = torch.float  # Tensor data type
        supervised_samples = {n for n in range(8)}
        unsupervised_samples = {n for n in range(8, 16)}
        dim_z = 3
        self.mesh = PoissonFEM.RectangularMesh(np.ones(lin_dim_rom) / lin_dim_rom)

        def origin(x):
            return np.abs(x[0]) < np.finfo(float).eps and np.abs(x[1]) < np.finfo(float).eps

        def ess_boundary_fun(x):
            return 0.0

        self.mesh.set_essential_boundary(origin, ess_boundary_fun)

        def domain_boundary(x):
            # unit square
            return np.abs(x[0]) < np.finfo(float).eps or np.abs(x[1]) < np.finfo(float).eps or \
                   np.abs(x[0]) > 1.0 - np.finfo(float).eps or np.abs(x[1]) > 1.0 - np.finfo(float).eps

        self.mesh.set_natural_boundary(domain_boundary)

        def flux(x):
            q = np.array([a[0] + a[2] * x[1], a[1] + a[2] * x[0]])
            return q

        # Specify right hand side and stiffness matrix
        rhs = PoissonFEM.RightHandSide(self.mesh)
        rhs.set_natural_rhs(self.mesh, flux)
        K = PoissonFEM.StiffnessMatrix(self.mesh)
        rhs.set_rhs_stencil(self.mesh, K)

        trainingData = dta.StokesData(supervised_samples, unsupervised_samples)
        trainingData.read_data()
        trainingData.reshape_microstructure_image()

        # define rom
        rom = ROM.ROM(self.mesh, K, rhs, trainingData.output_resolution ** 2)

        # finally set up model
        self.model = GenerativeSurrogate(rom, trainingData, dim_z=dim_z)


    def test_save_load(self):
        # do a couple of training steps, then save
        print('Doing a couple of training iterations...')
        self.model.fit(n_steps=2)
        print('...done.')
        print('Saving the model...')
        self.model.save()
        print('...done.')

        # load the model, then do a couple of training iterations to see if everything works
        print('Loading the model...')
        loaded_model = GenerativeSurrogate()
        loaded_model.load()
        print('...done.')

        print('Doing some more training iterations on the loaded model...')
        self.model.fit(n_steps=2)
        print('...done.')

        # finally, do some predictions and see if everything runs
        print('Doing some predictions...')
        test_samples = {n for n in range(0, 4)}
        testData = dta.StokesData(unsupervised_samples=test_samples)
        testData.read_data()
        # trainingData.plotMicrostruct(1)
        testData.reshape_microstructure_image()
        uf_pred, Z = self.model.predict(testData, max_iterations=200)
        print('...done.')









