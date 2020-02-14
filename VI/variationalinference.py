"""Module for stochastic variational inference"""

from torch import optim
import torch


class StochasticVariationalInference:
    def __init__(self, log_emp_dist, dim, n_data):
        self.log_emp_dist = log_emp_dist            # n_data outputs for every data point
        self.n_data = n_data
        self.dim = dim


class DiagGaussianSVI(StochasticVariationalInference):
    def __init__(self, log_emp_dist, dim, n_data=1):
        super(DiagGaussianSVI, self).__init__(log_emp_dist, dim, n_data)
        self.variationalDistribution = None

        self.loc = torch.zeros(self.n_data, self.dim, requires_grad=True)
        self.log_std = torch.zeros(self.n_data, self.dim, requires_grad=True)

        self.optimizer = optim.Adam([self.loc, self.log_std])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=1e4, verbose=True)
        self.min_lr = .01*self.optimizer.param_groups[0]['lr']
        self.max_iter = 1e8

    @property
    def vi_std(self):
        return torch.exp(self.log_std)

    def negative_elbo_sample(self):
        epsilon = torch.randn_like(self.loc)
        z = self.loc + torch.exp(self.log_std) * epsilon
        return torch.sum(self.log_emp_dist(z)) - torch.sum(self.log_std)

    def fit(self):
        converged = False
        iter = 0
        while not converged:
            # loss = self.autograd_elbo(self.loc, self.log_std)
            loss = self.negative_elbo_sample()
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


class FullRankGaussianSVI(StochasticVariationalInference):
    def __init__(self, log_emp_dist, dim, n_data):
        super(FullRankGaussianSVI, self).__init__(log_emp_dist, dim, n_data)
        loc = torch.zeros(n_data, self.dim, requires_grad=True)
        scale_tril = torch.eye(self.dim).unsqueeze(0).repeat(n_data, 1, 1).clone().detach().requires_grad_(True)
        self.variationalDistribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=loc, scale_tril=scale_tril)
        self.variationalDistribution = self.variationalDistribution.expand([n_data])

        self.optimizer = optim.Adam([loc, scale_tril])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=1e4, verbose=True)
        self.min_lr = .01*self.optimizer.param_groups[0]['lr']
        self.max_iter = 1e8

    def negative_elbo_sample(self):
        epsilon = torch.randn_like(self.variationalDistribution.loc)
        scale_tril = torch.tril(self.variationalDistribution.scale_tril)
        z = (self.variationalDistribution.loc + torch.bmm(scale_tril, epsilon.unsqueeze(2)).squeeze())
        return torch.sum(self.log_emp_dist(z) - torch.logdet(scale_tril))

    def fit(self):
        converged = False
        iter = 0
        while not converged:
            loss = self.negative_elbo_sample()
            self.variationalDistribution.loc.retain_grad()
            self.variationalDistribution.scale_tril.retain_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step(loss)
            if iter > self.max_iter:
                converged = True
                print(f"VI converged because max number of {self.max_iter} iterations reached")
            elif self.optimizer.param_groups[0]['lr'] < self.min_lr:
                converged = True
                print(f'VI converged because learning rate dropped below threshold {self.min_lr} (scheduler)')
            else:
                iter += 1









