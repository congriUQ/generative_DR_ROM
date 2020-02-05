"""Module for stochastic variational inference"""

from torch import optim
import torch


class StochasticVariationalInference:
    def __init__(self, log_emp_dist, log_emp_dist_grad, dim):
        self.log_emp_dist = log_emp_dist
        self.log_emp_dist_grad = log_emp_dist_grad
        self.dim = dim
        self.params = {}
        self.elbo_estimate = 0
        self.elbo_grad_estimate = torch.zeros(self.dim)  # w.r.t. z


class DiagGaussianSVI(StochasticVariationalInference):
    def __init__(self, log_emp_dist, log_emp_dist_grad, dim):
        super(DiagGaussianSVI, self).__init__(log_emp_dist, log_emp_dist_grad, dim)
        self.params['loc'] = torch.zeros(self.dim, requires_grad=True)
        self.params['log_std'] = torch.zeros(self.dim, requires_grad=True)

        self.autograd_elbo = self.get_autograd_elbo()

        self.optimizer = optim.Adam([self.params['loc'], self.params['log_std']])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=1e4, verbose=True)
        self.min_lr = .01*self.optimizer.param_groups[0]['lr']
        self.max_iter = 1e8

    @property
    def vi_std(self):
        return torch.exp(self.params['log_std'])

    def evaluate(self, z):
        """
        Evaluates the elbo for a single z only!
        """
        self.elbo_estimate = self.log_emp_dist(z) - sum(self.params['log_std'])

    def evaluate_grad(self, z):
        self.elbo_grad_estimate = self.log_emp_dist_grad(z)

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
                return self.elbo_estimate

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_p/dz
                z, epsilon, std = ctx.saved_tensors
                self.evaluate_grad(z)
                return self.elbo_grad_estimate, std*epsilon*self.elbo_grad_estimate - 1.0

        return AutogradElbo.apply

    def fit(self):
        converged = False
        iter = 0
        while not converged:
            loss = self.autograd_elbo(self.params['loc'], self.params['log_std'])
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
    def __init__(self, log_emp_dist, log_emp_dist_grad, dim):
        super(FullRankGaussianSVI, self).__init__(log_emp_dist, log_emp_dist_grad, dim)
        loc = torch.zeros(self.dim, requires_grad=True)
        scale_tril = torch.eye(self.dim, requires_grad=True)
        self.variationalDistribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=loc, scale_tril=scale_tril)

        self.autograd_elbo = self.get_autograd_elbo()

        self.optimizer = optim.Adam([loc, scale_tril])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=1e4, verbose=True)
        self.min_lr = .01*self.optimizer.param_groups[0]['lr']
        self.max_iter = 1e8

    def evaluate(self, z):
        """
        Evaluates the elbo for a single z only!
        """
        self.elbo_estimate = self.log_emp_dist(z) - torch.logdet(self.variationalDistribution.scale_tril)

    def evaluate_grad(self, z):
        # Gradient w.r.t. dependent variable z
        self.elbo_grad_estimate = self.log_emp_dist_grad(z)

    def get_autograd_elbo(self):
        """
        Creates an autograd function of elbo
        """

        class AutogradElbo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, mean, scale_tril):
                # The dependent variable epsilon is a torch.tensor with requires_grad=True;
                # it is a standard normal random variable
                epsilon = torch.randn_like(mean)
                z = mean + scale_tril@epsilon
                ctx.save_for_backward(z, epsilon, scale_tril)
                self.evaluate(z)
                return self.elbo_estimate

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_p/dz
                z, epsilon, scale_tril = ctx.saved_tensors
                self.evaluate_grad(z)
                return self.elbo_grad_estimate, \
                       torch.tril(torch.ger(self.elbo_grad_estimate, epsilon) - torch.inverse(scale_tril.T))

        return AutogradElbo.apply

    def fit(self):
        converged = False
        iter = 0
        while not converged:
            loss = self.autograd_elbo(self.variationalDistribution.loc, self.variationalDistribution.scale_tril)
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









