'''
Darcy flow reduced order model class
'''
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from matplotlib import pyplot as plt
import torch


class ROM:
    def __init__(self, mesh, stiffnessMatrix, rhs):
        self.mesh = mesh
        self.stiffnessMatrix = stiffnessMatrix
        self.rhs = rhs
        self.solution = PETSc.Vec().createSeq(mesh.nEq)
        self.adjoints = PETSc.Vec().createSeq(mesh.nEq)

    def solve(self, x):
        # x is a 1D torch.tensor of log conductivities/permeabilities
        lmbda = PETSc.Vec()
        lmbda.createWithArray(torch.exp(x))
        self.stiffnessMatrix.assemble(lmbda)
        self.rhs.assemble(lmbda)
        # self.stiffnessMatrix.solver.setOperators(self.stiffnessMatrix.matrix)
        self.stiffnessMatrix.solver.solve(self.rhs.vector, self.solution)

    def solveAdjoint(self, grad_output):
        # Call only after stiffness matrix has already been assembled!
        # grad_output is typically dlog_Pcf_du
        adjoint_rhs = PETSc.Vec()
        adjoint_rhs.createWithArray(grad_output)
        self.stiffnessMatrix.solver.solveTranspose(adjoint_rhs, self.adjoints)

    def plotSolution(self):
        solutionArray = np.zeros((self.mesh.nElX + 1, self.mesh.nElY + 1))
        X = solutionArray.copy()
        Y = solutionArray.copy()
        for vtx in self.mesh.vertices:
            X[vtx.rowIndex, vtx.colIndex] = vtx.coordinates[0]
            Y[vtx.rowIndex, vtx.colIndex] = vtx.coordinates[1]
            if vtx.isEssential:
                solutionArray[vtx.rowIndex, vtx.colIndex] = vtx.boundaryValue
            else:
                solutionArray[vtx.rowIndex, vtx.colIndex] = self.solution.array[vtx.equationNumber]

        plt.contourf(X, Y, solutionArray)

    def getAutogradFun(self):
        '''
        Creates an autograd function of ROM
        '''

        class fROM(torch.autograd.Function):

            @staticmethod
            def forward(ctx, X):
                # X is a torch.tensor with requires_grad=True
                # X are log diffusivities, i.e., they can be negative

                self.solve(X)
                return torch.tensor(self.solution.array)

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_Pcf_du
                self.solveAdjoint(grad_output)
                grad = PETSc.Vec()
                term0 = PETSc.Vec()
                term0.createWithArray(- torch.matmul(torch.matmul(self.stiffnessMatrix.globStiffGrad,
                                                                  torch.tensor(self.solution.array)).t(),
                                                     torch.tensor(self.adjoints.array)))
                self.rhs.rhsStencil.matMultTransposeAdd(self.adjoints, term0, grad)
                return torch.tensor(grad)

        return fROM.apply

