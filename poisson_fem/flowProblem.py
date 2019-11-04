"""Module for Stokes and Darcy (Poisson) flow problems"""


class FlowProblem:
    def __init__(self, a0=.0, ax=1.0, ay=.0, axy=.0):
        # Boundary conditions deduced from the pressure field p = a0 + ax*x = ay*y + axy*x*y,
        # u = - grad p = [-ax - axy*y, -ay - axy*x]
        self.a0 = a0
        self.ax = ax
        self.ay = ay
        self.axy = axy


class PoissonProblem(FlowProblem):
    def __init__(self, mesh, a0=.0, ax=1.0, ay=1.0, axy=0.0):
        super().__init__(a0, ax, ay, axy)
        self.mesh = mesh
