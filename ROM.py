'''
Darcy flow reduced order model class
'''
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from matplotlib import pyplot as plt


class ROM:
    def __init__(self, mesh, stiffnessMatrix, rhs, ksp):
        self.mesh = mesh
        self.stiffnessMatrix = stiffnessMatrix
        self.rhs = rhs
        self.solver = ksp
        self.solution = PETSc.Vec().createSeq(mesh.nEq)

    def solve(self, x):
        # x is a PETSc vector of conductivities/permeabilities
        self.stiffnessMatrix.assemble(x)
        self.rhs.assemble(x)
        self.solver.setOperators(self.stiffnessMatrix.matrix)
        self.solver.solve(self.rhs.vector, self.solution)

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
