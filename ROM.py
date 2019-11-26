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
    def __init__(self, mesh, stiffnessMatrix, rhs, out_dim):
        self.dtype = torch.float32

        self.mesh = mesh
        self.stiffnessMatrix = stiffnessMatrix
        self.rhs = rhs
        self.solution = PETSc.Vec().createSeq(mesh.n_eq)
        # including essential boundary conditions
        self.full_solution = PETSc.createSeq(mesh.n_vertices)
        # interpolated to fine scale solution space
        self.interpolated_solution = PETSc.Vec().createSeq(out_dim)

        self.adjoints = PETSc.Vec().createSeq(mesh.n_eq)

        # Preallocated PETSc vector storing the gradient
        self.grad = PETSc.Vec().createSeq(mesh.n_cells)

    def solve(self, lmbda):
        # lmbda is a 1D numpy array of !positive! conductivities/permeabilities
        lmbda = PETSc.Vec().createWithArray(lmbda)
        self.stiffnessMatrix.assemble(lmbda)
        self.rhs.assemble(lmbda)
        # self.stiffnessMatrix.solver.setOperators(self.stiffnessMatrix.matrix)
        self.stiffnessMatrix.solver.solve(self.rhs.vector, self.solution)

    def set_full_solution(self):
        # adds essential boundary conditions and solution of equation system
        # solve first!
        self.mesh.scatter_matrix(self.solution, self.mesh.essential_solution_vector, self.full_solution)

    def interpolate_to(self):
        # interpolates to fine scale solution
        self.mesh.interpolation_matrix.mult(self.full_solution, self.interpolated_solution)

    def solve_adjoint(self, grad_output):
        # Call only after stiffness matrix has already been assembled!
        # grad_output is typically dlog_Pcf_du
        adjoint_rhs = PETSc.Vec()
        adjoint_rhs.createWithArray(grad_output)
        self.stiffnessMatrix.solver.solveTranspose(adjoint_rhs, self.adjoints)

    def plot_solution(self):
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

    def get_autograd_fun(self):
        '''
        Creates an autograd function of ROM
        '''

        class AutogradROM(torch.autograd.Function):

            @staticmethod
            def forward(ctx, lmbda):
                # lmbda is a torch.tensor with requires_grad=True
                # lmbda are !positive! diffusivities

                self.solve(lmbda.detach().numpy())
                return torch.tensor(self.solution.array)

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_Pcf_du
                self.solve_adjoint(grad_output)
                term0 = PETSc.Vec().createWithArray(-torch.matmul(torch.matmul(self.stiffnessMatrix.glob_stiff_grad,
                            torch.tensor(self.solution.array, dtype=self.dtype)).t(),
                                                     torch.tensor(self.adjoints.array, dtype=self.dtype)))
                self.rhs.rhs_stencil.multTransposeAdd(self.adjoints, term0, self.grad)
                return torch.tensor(self.grad.array)

        return AutogradROM.apply

