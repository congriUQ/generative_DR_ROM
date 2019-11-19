#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib qt

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
dim_z = 100                            # Latent space dimension
dtype = torch.float                  # Tensor data type


# In[3]:


# Define mesh and boundary conditions
mesh = PoissonFEM.RectangularMesh(np.ones(lin_dim_rom)/lin_dim_rom)
# mesh.plot()

def origin(x):
    return np.abs(x[0]) < np.finfo(float).eps and np.abs(x[1]) < np.finfo(float).eps

def essBoundaryFun(x):
    return 0.0
mesh.setEssentialBoundary(origin, essBoundaryFun)

def domainBoundary(x):
    # unit square
    return np.abs(x[0]) < np.finfo(float).eps or np.abs(x[1]) < np.finfo(float).eps or             np.abs(x[0]) > 1.0 - np.finfo(float).eps or np.abs(x[1]) > 1.0 - np.finfo(float).eps
mesh.setNaturalBoundary(domainBoundary)

def flux(x):
    q = np.array([a[0] + a[2]*x[1], a[1] + a[2]*x[0]])
    return q


# In[4]:


#Spepify right hand side and stiffness matrix
#Define boundary flux field
rhs = PoissonFEM.RightHandSide(mesh)
rhs.setNaturalRHS(mesh, flux)
funSpace = PoissonFEM.FunctionSpace(mesh)
K = PoissonFEM.StiffnessMatrix(mesh, funSpace)
rhs.setRhsStencil(mesh, K)


# In[5]:


# Set up solver
ksp = PETSc.KSP().create()
ksp.setType('preonly')
precond = ksp.getPC()
precond.setType('cholesky')
ksp.setFromOptions() #???


# In[6]:


# define rom
rom = ROM.ROM(mesh, K, rhs, ksp)


# In[7]:


trainingData = dta.StokesData(range(512))
trainingData.readData(['IMG'])
# trainingData.plotMicrostruct(1)


# In[8]:


model = gs.GenerativeSurrogate(rom, trainingData, dim_z)


# In[9]:


steps = int(150)
for s in range(steps):
    print('step = ', s)
    batchSamples = torch.LongTensor(model.batchSizeN).random_(0, trainingData.nSamples)
    model.pfStep(batchSamples)


# In[10]:


model.plotInputReconstruction()


# In[11]:


f = plt.gcf()
f.suptitle('Untrained, N = 2048', fontsize=32, y=.7)


# In[12]:


trainingData.imgResolution**2*32*np.log(2)


# In[13]:


trainingData


# In[14]:


trainingData.microstructImg.nelement()


# In[15]:


X = torch.zeros(int(1e4), int(1e5))


# In[16]:


X.shape


# In[17]:


X.unsqueeze(2)
X = X.expand(int(1e4), int(1e5))


# In[18]:


X.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




