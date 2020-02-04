import unittest
import torch
import variationalinference as vi


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class SVIGauss2Gauss1D(unittest.TestCase):

    def test_gauss2gauss_1d(self):
        tol = 3e-2
        print('Testing Gauss to Gauss VI in 1D...')
        sigma_e = 2 * torch.ones(1)
        mu_e = -3 * torch.ones(1)

        # Note the positive sign. We want to maximize the elbo/minimize the loss.
        def log_emp_dist(x):
            return .5 * ((x - mu_e)**2.0)/(sigma_e**2)

        def log_emp_dist_grad(x):
            return (x - mu_e)/sigma_e**2

        svi = vi.DiagGaussianSVI(log_emp_dist, log_emp_dist_grad, 1)
        svi.fit()

        print('True mean == ', mu_e)
        print('VI mean == ', svi.vi_mean)
        print('True std == ', sigma_e)
        print('VI std == ', svi.vi_std)
        # 3% deviation is allowed
        self.assertLess(abs(mu_e - svi.vi_mean)/(abs(mu_e) + 1e-6), tol)
        self.assertLess(abs(sigma_e - svi.vi_std)/(abs(sigma_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 1D test passed.')


class SVIGauss2Gauss2D(unittest.TestCase):

    def test_gauss2gauss_2d(self):
        # 3 percent error tolerance
        tol = 3e-2
        print('Testing Gauss to Gauss VI in 2D...')
        sigma_e = torch.tensor([2.0, 3.0])
        mu_e = torch.tensor([-2.0, 3.0])

        def log_emp_dist(x):
            return .5*sum(((x - mu_e)**2.0)/(sigma_e**2))

        def log_emp_dist_grad(x):
            return (x - mu_e)/sigma_e**2

        svi = vi.DiagGaussianSVI(log_emp_dist, log_emp_dist_grad, 2)
        svi.fit()

        print('True mean == ', mu_e)
        print('VI mean == ', svi.vi_mean)
        print('True std == ', sigma_e)
        print('VI std == ', svi.vi_std)
        self.assertLess(abs(mu_e - svi.vi_mean)/(abs(mu_e) + 1e-6), tol)
        self.assertLess(abs(sigma_e - svi.vi_std)/(abs(sigma_e) + 1e-6), tol)
        print('... Gauss to Gauss VI in 2D test passed.')


if __name__ == '__main__':
    unittest.main()
