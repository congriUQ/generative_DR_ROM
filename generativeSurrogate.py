from poisson_fem import PoissonFEM
import ROM
import GenerativeSurrogate as gs
import Data as dta
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Script to train the model. First input is number of supervised,'
                                             ' second input is number of unsupervised samples')
parser.add_argument(action="store", dest="n_supervised", type=int)
parser.add_argument(action="store", dest="n_unsupervised", type=int)
parser.add_argument(action="store", dest="dim_z", type=int)
parsed_args = parser.parse_args()

# Some parameters
lin_dim_rom = 4                      # Linear number of rom elements
a = np.array([1, 1, 0])              # Boundary condition function coefficients
dtype = torch.float                  # Tensor data type
n_supervised = parsed_args.n_supervised
n_unsupervised = parsed_args.n_unsupervised
supervised_samples = {n for n in range(n_supervised)}
unsupervised_samples = {n for n in range(n_supervised, n_supervised + n_unsupervised)}

# Define mesh and boundary conditions
mesh = PoissonFEM.RectangularMesh(np.ones(lin_dim_rom)/lin_dim_rom)
# mesh.plot()


def origin(x):
    return np.abs(x[0]) < np.finfo(float).eps and np.abs(x[1]) < np.finfo(float).eps


def ess_boundary_fun(x):
    return 0.0
mesh.set_essential_boundary(origin, ess_boundary_fun)


def domain_boundary(x):
    # unit square
    return np.abs(x[0]) < np.finfo(float).eps or np.abs(x[1]) < np.finfo(float).eps or \
            np.abs(x[0]) > 1.0 - np.finfo(float).eps or np.abs(x[1]) > 1.0 - np.finfo(float).eps
mesh.set_natural_boundary(domain_boundary)


def flux(x):
    q = np.array([a[0] + a[2]*x[1], a[1] + a[2]*x[0]])
    return q


#Specify right hand side and stiffness matrix
rhs = PoissonFEM.RightHandSide(mesh)
rhs.set_natural_rhs(mesh, flux)
K = PoissonFEM.StiffnessMatrix(mesh)
rhs.set_rhs_stencil(mesh, K)


trainingData = dta.StokesData(supervised_samples, unsupervised_samples)
trainingData.read_data()
# trainingData.plotMicrostruct(1)
trainingData.reshape_microstructure_image()

# define rom
rom = ROM.ROM(mesh, K, rhs, trainingData.output_resolution**2)
model = gs.GenerativeSurrogate(rom, trainingData, dim_z=parsed_args.dim_z)

# optimize lambdac first
for n in range(model.data.n_supervised_samples):
    print('sample == ', n)
    model.log_lambdac_mean[n].max_iter = 3e4
    model.log_lambdac_mean[n].converge(model, model.data.n_supervised_samples, mode=n)

model.fit(n_steps=50, with_precisions=False, z_iterations=200, thetac_iterations=10000, lambdac_iterations=500)
model.fit(n_steps=int(1e6), save_iterations=5, thetac_iterations=5000, thetaf_iterations=50, z_iterations=100,
          with_precisions=True)




