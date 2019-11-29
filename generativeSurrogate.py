import matplotlib.pyplot as plt
from poisson_fem import PoissonFEM
import ROM
import GenerativeSurrogate as gs
import Data as dta
import numpy as np
import torch

# Some parameters
lin_dim_rom = 4                      # Linear number of rom elements
a = np.array([1, 1, 0])              # Boundary condition function coefficients
dim_z = 20                            # Latent space dimension
dtype = torch.float                  # Tensor data type
supervised_samples = {n for n in range(16)}
unsupervised_samples = {n for n in range(16, 500)}

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


# Specify right hand side and stiffness matrix
rhs = PoissonFEM.RightHandSide(mesh)
rhs.set_natural_rhs(mesh, flux)
funSpace = PoissonFEM.FunctionSpace(mesh)
K = PoissonFEM.StiffnessMatrix(mesh, funSpace)
rhs.set_rhs_stencil(mesh, K)

trainingData = dta.StokesData(supervised_samples, unsupervised_samples)
trainingData.read_data()
# trainingData.plotMicrostruct(1)
trainingData.reshape_microstructure_image()

# define rom and set up model
rom = ROM.ROM(mesh, K, rhs, trainingData.output_resolution**2)
model = gs.GenerativeSurrogate(rom, trainingData, dim_z)
model.fit(n_steps=100)


