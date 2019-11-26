#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
from poisson_fem import PoissonFEM
import ROM
import GenerativeSurrogate as gs
import Data as dta
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as lg
import time
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import torch
from torch import optim


# In[2]:


# Some fixed parameters
lin_dim_rom = 4                      # Linear number of rom elements
a = np.array([1, 2, 3])              # Boundary condition function coefficients
dim_z = 30                            # Latent space dimension
dtype = torch.float                  # Tensor data type


# In[3]:


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
    return np.abs(x[0]) < np.finfo(float).eps or np.abs(x[1]) < np.finfo(float).eps or             np.abs(x[0]) > 1.0 - np.finfo(float).eps or np.abs(x[1]) > 1.0 - np.finfo(float).eps
mesh.set_natural_boundary(domain_boundary)

def flux(x):
    q = np.array([a[0] + a[2]*x[1], a[1] + a[2]*x[0]])
    return q


# In[4]:


# Set up solver
ksp = PETSc.KSP().create()
ksp.setType('preonly')
precond = ksp.getPC()
precond.setType('cholesky')
ksp.setFromOptions() #???


# In[5]:


#Spepify right hand side and stiffness matrix
#Define boundary flux field
rhs = PoissonFEM.RightHandSide(mesh)
rhs.set_natural_rhs(mesh, flux)
funSpace = PoissonFEM.FunctionSpace(mesh)
K = PoissonFEM.StiffnessMatrix(mesh, funSpace, ksp)
rhs.set_rhs_stencil(mesh, K)


# In[6]:


# define rom
rom = ROM.ROM(mesh, K, rhs)


# In[7]:


trainingData = dta.StokesData(range(256))
trainingData.read_data(['IMG'])
# trainingData.plotMicrostruct(1)
trainingData.reshape_microstructure_image()


# In[8]:


model = gs.GenerativeSurrogate(rom, trainingData, dim_z)


# In[9]:


steps = int(1)
for s in range(steps):
    print('step = ', s)
    batch_samples_pf = torch.multinomial(torch.ones(trainingData.n_samples_in), model.batch_size_N_pf)
#     batch_samples_pc = torch.multinomial(torch.ones(trainingData.n_samples_out), model.batch_size_N_pc)
    model.opt_latent_dist_step()
    model.pf_step(batch_samples_pf)
#     model.pc_step(batch_samples_pc)
    


# In[10]:


model.plot_input_reconstruction()
f = plt.gcf()
f.suptitle('Untrained, N = 1184', fontsize=32, y=.7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


plt.figure()
trainingData.plot_microstruct(0)


# In[12]:


predfun = rom.get_autograd_fun()
u_pred = predfun(torch.ones(16, requires_grad=True))


# In[13]:


u = torch.ones(24)


# In[14]:


loss = (u - u_pred).pow(2).sum()


# In[ ]:


loss.backward()


# In[15]:


ts = torch.tensor(rom.solution.array)


# In[18]:


rom.adjoints.array


# In[ ]:





# In[ ]:





# In[16]:


rom.rhs.rhs_stencil.multTransposeAdd(rom.adjoints)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




