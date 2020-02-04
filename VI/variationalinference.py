"""Module for stochastic variational inference"""

from torch import optim
import torch


class StochasticVariationalInference:
    def __init__(self):
        pass


class DiagGaussianSVI:
    def __init__(self, log_emp_dist, log_emp_dist_grad, dim):
        self.log_emp_dist = log_emp_dist
        self.log_emp_dist_grad = log_emp_dist_grad
        self.vi_mean = torch.zeros(dim, requires_grad=True)
        self.vi_log_std = torch.zeros(dim, requires_grad=True)
        self.estimate = 0
        self.grad_estimate = torch.zeros(dim)   # w.r.t. z

        self.autograd_elbo = self.get_autograd_elbo()

        self.optimizer = optim.Adam([self.vi_mean, self.vi_log_std])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=1e5, verbose=True)
        self.min_lr = .01*self.optimizer.param_groups[0]['lr']
        self.max_iter = 1e8

    @property
    def vi_std(self):
        return torch.exp(self.vi_log_std)

    def evaluate(self, z):
        """
        Evaluates the elbo for a single z only!
        """
        self.estimate = self.log_emp_dist(z) - sum(self.vi_log_std)

    def evaluate_grad(self, z):
        self.grad_estimate = self.log_emp_dist_grad(z)

    def get_autograd_elbo(self):
        """
        Creates an autograd function of elbo
        """

        class AutogradElbo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, mean, log_std):
                # The dependent variable epsilon is a torch.tensor with requires_grad=True;
                # it is a standard normal random variable
                epsilon = torch.randn_like(log_std)
                std = torch.exp(log_std)
                z = mean + std*epsilon
                ctx.save_for_backward(z, epsilon, std)
                self.evaluate(z)
                return self.estimate

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_p/dz
                z, epsilon, std = ctx.saved_tensors
                self.evaluate_grad(z)
                return self.grad_estimate, std*epsilon*self.grad_estimate - 1.0

        return AutogradElbo.apply

    def fit(self):
        converged = False
        iter = 0
        while not converged:
            loss = self.autograd_elbo(self.vi_mean, self.vi_log_std)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step(loss)
            if iter > self.max_iter:
                converged = True
                print('VI converged because max number of iterations reached')
            elif self.optimizer.param_groups[0]['lr'] < self.min_lr:
                converged = True
                print('VI converged because learning rate dropped below threshold (scheduler)')
            else:
                iter += 1


###########################################################################
# Tests
###########################################################################
import unittest


class SVITestCase(unittest.TestCase):

    def gauss2gauss_1d(self):
        sigma_e = 2 * torch.ones(1)
        mu_e = -3 * torch.ones(1)
        def log_emp_dist(x):
            return .5 * (1 / sigma_e ** 2) * (x - mu_e) ** 2

        def log_emp_dist_grad(x):
            return (x - mu_e) / sigma_e ** 2

        svi = DiagGaussianSVI(log_emp_dist, log_emp_dist_grad, 1)
        svi.fit()

        print("True mean == ", mu_e)
        print("SVI mean == ", svi.vi_mean)
        print("True std == ", sigma_e)
        print("SVI std == ", svi.vi_std)










