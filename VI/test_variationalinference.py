import unittest
import torch
import variationalinference as vi


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class SVIGauss2Gauss1D(unittest.TestCase):
    """
    This test fits a 1D Gaussian to a 1D Gaussian
    """

    def test_gauss2gauss_1d(self):
        tol = 3e-2
        print('Testing Gauss to Gauss VI in 1D...')
        sigma_e = 2 * torch.ones(1)
        mu_e = -3 * torch.ones(1)

        # Note the positive sign. We want to maximize the elbo/minimize the loss.
        def log_emp_dist(x):
            return .5 * ((x - mu_e)**2.0)/(sigma_e**2)

        svi = vi.DiagGaussianSVI(log_emp_dist, dim=1)
        svi.fit()

        print('True mean == ', mu_e.data)
        print('VI mean == ', svi.loc.data)
        print('True std == ', sigma_e.data)
        print('VI std == ', svi.vi_std.data)
        # 3% deviation is allowed
        self.assertLess(abs(mu_e - svi.loc)/(abs(mu_e) + 1e-6), tol)
        self.assertLess(abs(sigma_e - svi.vi_std)/(abs(sigma_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 1D test passed.')


class SVIGauss2Gauss1D_batched(unittest.TestCase):
    """
    This test fits two 1D Gaussians to two 1D Gaussians in a batched fashion
    """

    def test_gauss2gauss_1d(self):
        tol = 3e-2
        print('Testing Gauss to Gauss VI in 1D in a batched fashion for 2 distributions...')
        mu_emp = torch.tensor([2.0, -3.0]).unsqueeze(1)  # shape = n_batch_samples x 1
        sigma_emp = torch.tensor([1.0, 3.0]).unsqueeze(1)

        # Note the positive sign. We want to maximize the elbo/minimize the loss.
        def log_emp_dist(x):
            return .5 * ((x - mu_emp)**2.0)/(sigma_emp**2)

        svi = vi.DiagGaussianSVI(log_emp_dist, dim=1, n_data=mu_emp.numel())
        svi.fit()

        print('True means == ', mu_emp.data)
        print('VI means == ', svi.loc.data)
        print('True stds == ', sigma_emp.data)
        print('VI stds == ', svi.vi_std.data)
        # 3% deviation is allowed
        assert all((mu_emp - svi.loc).abs()/(mu_emp.abs() + 1e-6) < tol)
        assert all((sigma_emp - svi.vi_std).abs()/(sigma_emp.abs() + 1e-6) < tol)
        print('... Gauss to Gauss VI in 1D test passed.')


class SVIDiagGauss2Gauss2D(unittest.TestCase):

    def test_gauss2gauss_2d(self):
        # 3 percent error tolerance
        tol = 3e-2
        print('Testing diagonal Gauss to Gauss VI in 2D...')
        sigma_e = torch.tensor([2.0, 3.0])
        mu_e = torch.tensor([-2.0, 3.0])

        def log_emp_dist(x):
            return .5*sum(((x - mu_e)**2.0)/(sigma_e**2))

        svi = vi.DiagGaussianSVI(log_emp_dist, 2)
        svi.fit()

        print('True mean == ', mu_e.data)
        print('VI mean == ', svi.loc.data)
        print('True std == ', sigma_e.data)
        print('VI std == ', svi.vi_std.data)
        self.assertLess(torch.norm(mu_e - svi.loc)/(torch.norm(mu_e) + 1e-6), tol)
        self.assertLess(torch.norm(sigma_e - svi.vi_std)/(torch.norm(sigma_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 2D test passed.')


class SVIDiagGauss2Gauss2D_batched(unittest.TestCase):

    def test_gauss2gauss_2d(self):
        # 3 percent error tolerance
        tol = 3e-2
        print('Testing batched diagonal Gauss to Gauss VI in 2D for 2 distributions...')
        sigma_e = torch.tensor([[1.0, 5.0], [2.0, 3.0]])
        mu_e = torch.tensor([[1.0, -4.0], [-2.0, 3.0]])

        def log_emp_dist(x):
            return .5*torch.sum(((x - mu_e)**2.0)/(sigma_e**2))

        svi = vi.DiagGaussianSVI(log_emp_dist, dim=2, n_data=mu_e.shape[0])
        svi.fit()

        print('True mean == ', mu_e.data)
        print('VI mean == ', svi.loc.data)
        print('True std == ', sigma_e.data)
        print('VI std == ', svi.vi_std.data)
        self.assertLess(torch.norm(mu_e - svi.loc)/(torch.norm(mu_e) + 1e-6), tol)
        self.assertLess(torch.norm(sigma_e - svi.vi_std)/(torch.norm(sigma_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 2D test passed.')


class SVIMVGauss2Gauss2D(unittest.TestCase):

    def test_gauss2gauss_2d(self):
        # 3 percent error tolerance
        tol = 3e-2
        print('Testing multivariate Gauss to Gauss VI in 2D...')
        L_e = torch.eye(2)
        L_e[1, 0] = .5
        Sigma_e = L_e @ L_e.T
        mu_e = torch.tensor([-2.0, 3.0])

        def log_emp_dist(x):
            # Can be done more efficiently using the existing LU decompusition
            rhs = (x - mu_e).unsqueeze(2)
            # return .5*(x - mu_e).T @ torch.cholesky_solve(rhs, L_e).squeeze()
            return .5 * torch.bmm(torch.transpose(rhs, 2, 1), torch.cholesky_solve(rhs, L_e)).squeeze()

        svi = vi.FullRankGaussianSVI(log_emp_dist, dim=2, n_data=1)
        svi.fit()

        print('True mean == ', mu_e)
        print('VI mean == ', svi.variationalDistribution.loc.data)
        print('True Sigma == ', Sigma_e)
        print('VI Sigma == ', svi.variationalDistribution.covariance_matrix.data)
        print('True L == ', L_e)
        print('VI L == ', svi.variationalDistribution.scale_tril.data)
        self.assertLess(torch.norm(mu_e - svi.variationalDistribution.loc)/(torch.norm(mu_e) + 1e-6), tol)
        self.assertLess(torch.norm(svi.variationalDistribution.scale_tril - L_e)/(torch.norm(L_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 2D test passed.')


class SVIMVGauss2Gauss2D_batched(unittest.TestCase):

    def test_gauss2gauss_2d(self):
        # 3 percent error tolerance
        tol = 3e-2
        print('Testing multivariate Gauss to Gauss VI in 2D in batched version...')
        L_e = torch.eye(2)
        L_e[1, 0] = .5
        L_e = L_e.unsqueeze(0).repeat(3, 1, 1)
        L_e[1, 1, 1] = 2.0
        Sigma_e = L_e @ torch.transpose(L_e, 1, 2)
        mu_e = torch.tensor([[-2.0, 3.0], [4.0, -2.0], [-7.0, 7.0]])

        def log_emp_dist(x):
            # Can be done more efficiently using the existing LU decompusition
            rhs = (x - mu_e).unsqueeze(2)
            return .5*torch.bmm(torch.transpose(rhs, 1, 2), torch.cholesky_solve(rhs, L_e)).squeeze()

        svi = vi.FullRankGaussianSVI(log_emp_dist, dim=2, n_data=mu_e.shape[0])
        svi.fit()

        print('True mean == ', mu_e)
        print('VI mean == ', svi.variationalDistribution.loc.data)
        print('True Sigma == ', Sigma_e)
        print('VI Sigma == ', svi.variationalDistribution.covariance_matrix.data)
        print('True L == ', L_e)
        print('VI L == ', svi.variationalDistribution.scale_tril.data)
        self.assertLess(torch.norm(mu_e - svi.variationalDistribution.loc)/(torch.norm(mu_e) + 1e-6), tol)
        self.assertLess(torch.norm(svi.variationalDistribution.scale_tril - L_e)/(torch.norm(L_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 2D batched version test passed.')


if __name__ == '__main__':
    unittest.main()
