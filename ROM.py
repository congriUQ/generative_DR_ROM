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
        # Contains only natural degrees of freedom -- solution to equation system
        self.solution = PETSc.Vec().createSeq(mesh.n_eq)
        # including essential boundary conditions
        self.full_solution = PETSc.Vec().createSeq(mesh.n_vertices)
        # interpolated to fine scale solution space
        self.interpolated_solution = PETSc.Vec().createSeq(out_dim)

        self.adjoints = PETSc.Vec().createSeq(mesh.n_eq)

        # Preallocated PETSc vectors storing the gradients
        # Grad of rom w.r.t. lambda_c
        self.grad = PETSc.Vec().createSeq(mesh.n_cells)
        # outer gradient of loss w.r.t. uf_pred
        self.grad_uf_pred = PETSc.Vec().createSeq(out_dim)
        # outer gradient w.r.t. uc including essential boundary conditions
        self.grad_uc_full = PETSc.Vec().createSeq(mesh.n_vertices)
        # outer gradient w.r.t. degrees of freedom uc
        self.grad_uc = PETSc.Vec().createSeq(mesh.n_eq)

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
        self.mesh.scatter_matrix.multAdd(self.solution, self.mesh.essential_solution_vector, self.full_solution)

    def interpolate_to_fine(self):
        # interpolates to fine scale solution
        self.mesh.interpolation_matrix.mult(self.full_solution, self.interpolated_solution)

    def solve_adjoint(self, grad_output):
        # Call only after stiffness matrix has already been assembled!
        # grad_output is typically dlog_Pcf_du
        self.stiffnessMatrix.solver.solveTranspose(grad_output, self.adjoints)

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

                # scatter essential boundary conditions
                self.set_full_solution()
                self.interpolate_to_fine()
                return torch.tensor(self.interpolated_solution.array, dtype=self.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                # grad_output = d_log_Pcf_duf_pred, uf_pred = W_cf * u_c
                self.grad_uf_pred.array = grad_output.detach().numpy()
                # from grad w.r.t. uf_pred to grad w.r.t. uc_full (with b.c.)
                self.mesh.interpolation_matrix.multTranspose(self.grad_uf_pred, self.grad_uc_full)
                # from grad w.r.t. uc_full (with b.c.) to grad w.r.t. uc (without bc)
                self.mesh.scatter_matrix.multTranspose(self.grad_uc_full, self.grad_uc)

                self.solve_adjoint(self.grad_uc)
                # can this be optimized using PETSc only?
                term0 = PETSc.Vec().createWithArray(-torch.matmul(torch.matmul(self.stiffnessMatrix.glob_stiff_grad,
                            torch.tensor(self.solution.array, dtype=self.dtype)).t(),
                                                     torch.tensor(self.adjoints.array, dtype=self.dtype)))
                self.rhs.rhs_stencil.multTransposeAdd(self.adjoints, term0, self.grad)

                # remove the unsqueeze once batched computation is implemented!!!
                return torch.tensor(self.grad.array).unsqueeze(0)

        return AutogradROM.apply

