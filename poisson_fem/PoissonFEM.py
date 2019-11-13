'''Poisson FEM base class'''
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sps
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from scipy.integrate import quad


class Mesh:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.cells = []
        self._nVertices = 0
        self._nEdges = 0
        self.nCells = 0
        self.nEq = None

    def createVertex(self, coordinates, globalVertexNumber=None, rowIndex=None, colIndex=None):
        # Creates a vertex and appends it to vertex list
        vtx = Vertex(coordinates, globalVertexNumber, rowIndex, colIndex)
        self.vertices.append(vtx)
        self._nVertices += 1

    def createEdge(self, vtx0, vtx1):
        # Creates an edge between vertices specified by vertex indices and adds it to edge list
        edg = Edge(vtx0, vtx1)
        self.edges.append(edg)

        # Link edges to vertices
        vtx0.addEdge(edg)
        vtx1.addEdge(edg)

        self._nEdges += 1

    def createCell(self, vertices, edges, number=None):
        # Creates a cell and appends it to cell list
        cll = Cell(vertices, edges, number)
        self.cells.append(cll)

        # Link cell to vertices
        for vtx in vertices:
            vtx.addCell(cll)

        # Link cell to edges
        for edg in edges:
            edg.addCell(cll)

        self.nCells += 1

    def setEssentialBoundary(self, locationIndicatorFun, valueFun):
        # locationIndicatorFun is the indicator function to the essential boundary
        equationNumber = 0
        for vtx in self.vertices:
            if locationIndicatorFun(vtx.coordinates):
                vtx.isEssential = True
                vtx.boundaryValue = valueFun(vtx.coordinates)
                for cll in vtx.cells:
                    cll.containsEssentialVertex = True
            else:
                vtx.equationNumber = equationNumber
                equationNumber += 1
        self.nEq = equationNumber

    def setNaturalBoundary(self, locationIndicatorFun):
        # locationIndicatorFun is the indicator function to the natural boundary
        # locationIndicatorFun should include pointwise essential boundaries, i.e., for essential boundary at
        # x = 0, y = 0, locationIndicatorFun should indicate the whole domain boundary
        for edg in self.edges:
            if locationIndicatorFun(edg.vertices[0].coordinates) and locationIndicatorFun(edg.vertices[1].coordinates):
                # Both edge vertices are on natural boundary, i.e., edge is natural boundary
                edg.isNatural = True

    def plot(self):
        for vtx in self.vertices:
            if vtx is not None:
                vtx.plot()

        n = 0
        for edg in self.edges:
            if edg is not None:
                edg.plot(n)
                n += 1

        # plot cell number into cell
        for cll in self.cells:
            if cll is not None:
                plt.text(cll.centroid[0], cll.centroid[1], str(cll.number))

        plt.xlabel('x')
        plt.ylabel('y')


class RectangularMesh(Mesh):
    # Assuming a unit square domain
    def __init__(self, gridX, gridY=None):
        # Grid vectors gridX, gridY give the edge lengths

        super().__init__()
        if gridY is None:
            # assume square mesh
            gridY = gridX

        # Create vertices
        xCoord = np.concatenate((np.array([.0]), np.cumsum(gridX)))
        yCoord = np.concatenate((np.array([.0]), np.cumsum(gridY)))
        n = 0
        rowIndex = 0
        for y in yCoord:
            colIndex = 0
            for x in xCoord:
                self.createVertex(np.array([x, y]), globalVertexNumber=n, rowIndex=rowIndex, colIndex=colIndex)
                n += 1
                colIndex += 1
            rowIndex += 1

        # Create edges
        nx = len(gridX) + 1
        ny = len(gridY) + 1
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x + y*nx].coordinates[0] > 0:
                    self.createEdge(self.vertices[x + y*nx - 1], self.vertices[x + y*nx])
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x + y*nx].coordinates[1] > 0:
                    self.createEdge(self.vertices[x + (y - 1)*nx], self.vertices[x + y*nx])

        # Create cells
        nx -= 1
        ny -= 1
        n = 0   # cell index
        for y in range(ny):
            for x in range(nx):
                vtx = [self.vertices[x + y*(nx + 1)], self.vertices[x + y*(nx + 1) + 1],
                       self.vertices[x + (y + 1)*(nx + 1) + 1], self.vertices[x + (y + 1)*(nx + 1)]]
                edg = [self.edges[n], self.edges[nx*(ny + 1) + n + y + 1], self.edges[n + nx],
                       self.edges[nx*(ny + 1) + n + y]]
                self.createCell(vtx, edg, number=n)
                n += 1

        # minor stuff
        self.nElX = len(gridX)
        self.nElY = len(gridY)
        self.nEl = self.nElX*self.nElY
        # self.shapeFunGradients = None
        # self.locStiffGrad = None
        # self.gradGlobStiff = []
        # self.globStiffStencil = None
        #
        # # Stiffness sparsity pattern
        # self.K_nnz = None
        # self.Kvec_nonzero = None
        # self.K_indptr = None
        # self.K_indices = None


class Cell:
    def __init__(self, vertices, edges, number=None):
        # Vertices and edges must be sorted according to local vertex/edge number!
        self.vertices = vertices
        self.edges = edges
        self.number = number
        self.centroid = []
        self.computeCentroid()
        self.surface = self.edges[0].length*self.edges[1].length
        self.containsEssentialVertex = False

    def computeCentroid(self):
        # Compute cell centroid
        self.centroid = np.zeros(2)
        for vtx in self.vertices:
            self.centroid += vtx.coordinates
        self.centroid /= len(self.vertices)

    def deleteEdges(self, indices):
        # Deletes edges according to indices by setting them to None
        for i in indices:
            self.edges[i] = None

    def inside(self, x):
        # Checks if point x is inside of cell
        return (self.vertices[0].coordinates[0] - np.finfo(float).eps < x[0]
                <= self.vertices[2].coordinates[0] + np.finfo(float).eps and
                self.vertices[0].coordinates[1] - np.finfo(float).eps < x[1]
                <= self.vertices[2].coordinates[1] + np.finfo(float).eps)


class Edge:
    def __init__(self, vtx0, vtx1):
        self.vertices = [vtx0, vtx1]
        self.cells = []
        self.length = np.linalg.norm(vtx0.coordinates - vtx1.coordinates)
        self.isNatural = False       # True if edge is on natural boundary

    def addCell(self, cell):
        self.cells.append(cell)

    def plot(self, n):
        plt.plot([self.vertices[0].coordinates[0], self.vertices[1].coordinates[0]],
                 [self.vertices[0].coordinates[1], self.vertices[1].coordinates[1]], linewidth=.5, color='r')
        plt.text(.5*(self.vertices[0].coordinates[0] + self.vertices[1].coordinates[0]),
                 .5*(self.vertices[0].coordinates[1] + self.vertices[1].coordinates[1]), str(n), color='r')


class Vertex:
    def __init__(self, coordinates=np.zeros((2, 1)), globalNumber=None, rowIndex=None, colIndex=None):
        self.coordinates = coordinates
        self.cells = []
        self.edges = []
        self.isEssential = False    # False for natural, True for essential vertex
        self.boundaryValue = None    # Value for essential boundary
        self.equationNumber = None   # Equation number of dof belonging to vertex
        self.globalNumber = globalNumber
        self.rowIndex = rowIndex
        self.colIndex = colIndex

    def addCell(self, cell):
        self.cells.append(cell)

    def addEdge(self, edge):
        self.edges.append(edge)

    def plot(self):
        p = plt.plot(self.coordinates[0], self.coordinates[1], 'bx', linewidth=2.0, markersize=8.0)
        plt.text(self.coordinates[0], self.coordinates[1], str(self.globalNumber), color='b')


class FunctionSpace:
    # Only bilinear is implemented!
    def __init__(self, mesh, typ='bilinear'):
        self.shapeFunGradients = None
        self.getShapeFunGradMat(mesh)

    def getShapeFunGradMat(self, mesh):
        # Computes shape function gradient matrices B, see Fish & Belytshko

        # Gauss quadrature points
        xi0 = -1.0 / np.sqrt(3)
        xi1 = 1.0 / np.sqrt(3)

        self.shapeFunGradients = np.empty((8, 4, mesh.nEl))
        for e in range(mesh.nEl):
            # short hand notation
            x0 = mesh.cells[e].vertices[0].coordinates[0]
            x1 = mesh.cells[e].vertices[1].coordinates[0]
            y0 = mesh.cells[e].vertices[0].coordinates[1]
            y3 = mesh.cells[e].vertices[3].coordinates[1]

            # coordinate transformation
            xI = .5 * (x0 + x1) + .5 * xi0 * (x1 - x0)
            xII = .5 * (x0 + x1) + .5 * xi1 * (x1 - x0)
            yI = .5 * (y0 + y3) + .5 * xi0 * (y3 - y0)
            yII = .5 * (y0 + y3) + .5 * xi1 * (y3 - y0)

            # B matrices for bilinear shape functions
            B0 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xI - x1, x0 - xI, xI - x0, x1 - xI]])
            B1 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B2 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B3 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xI - x1, x0 - xI, xI - x0, x1 - xI]])

            # Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence we take the
            # additional factor of sqrt(A)/2 onto B
            self.shapeFunGradients[:, :, e] = (1 / (2 * np.sqrt(mesh.cells[e].surface))) * np.concatenate(
                (B0, B1, B2, B3))


class StiffnessMatrix:
    def __init__(self, mesh, funSpace):
        self.mesh = mesh
        self.funSpace = funSpace
        self.rangeCells = range(mesh.nCells)  # For assembly. more efficient if only allocated once?

        self.locStiffGrad = None
        self.globStiffGrad = []
        self.globStiffStencil = None

        # Stiffness sparsity pattern
        self.nnz = None             # number of nonzero components
        self.vec_nonzero = None     # nonzero component indices of flattened stiffness matrix
        self.indptr = None          # csr indices
        self.indices = None

        # Pre-computations
        self.compGlobStiffStencil()
        self.compSparsityPattern()

        # Pre-allocations
        self.matrix = PETSc.Mat().createAIJ(size=(mesh.nEq, mesh.nEq), nnz=self.nnz)
        # self.conductivity = PETSc.Vec().createSeq(mesh.nCells)     # permeability/diffusivity vector
        self.assemblyVector = PETSc.Vec().createSeq(mesh.nEq**2)      # For quick assembly with matrix vector product

    def compEquationIndices(self):
        # Compute equation indices for direct assembly of stiffness matrix
        equationIndices0 = np.array([], dtype=np.uint32)
        equationIndices1 = np.array([], dtype=np.uint32)
        locIndices0 = np.array([], dtype=np.uint32)
        locIndices1 = np.array([], dtype=np.uint32)
        cllIndex = np.array([], dtype=np.uint32)
        for cll in self.mesh.cells:
            equations = np.array([], dtype=np.uint32)
            eqVertices = np.array([], dtype=np.uint32)
            i = 0
            for vtx in cll.vertices:
                if vtx.equationNumber is not None:
                    equations = np.append(equations, np.array([vtx.equationNumber], dtype=np.uint32))
                    eqVertices = np.append(eqVertices, np.array([i], dtype=np.uint32))
                i += 1
            eq0, eq1 = np.meshgrid(equations, equations)
            vtx0, vtx1 = np.meshgrid(eqVertices, eqVertices)
            equationIndices0 = np.append(equationIndices0, eq0.flatten())
            equationIndices1 = np.append(equationIndices1, eq1.flatten())
            locIndices0 = np.append(locIndices0, vtx0.flatten())
            locIndices1 = np.append(locIndices1, vtx1.flatten())
            cllIndex = np.append(cllIndex, cll.number*np.ones_like(vtx0.flatten()))
        kIndex = np.ravel_multi_index((locIndices1, locIndices0, cllIndex), (4, 4, self.mesh.nCells), order='F')
        return [equationIndices0, equationIndices1], kIndex

    def compLocStiffGrad(self):
        # Compute local stiffness matrix gradients w.r.t. diffusivities
        if self.funSpace.shapeFunGradients is None:
            self.funSpace.getShapeFunGradMat()

        self.locStiffGrad = self.mesh.nEl * [np.empty((4, 4))]
        for e in range(self.mesh.nEl):
            self.locStiffGrad[e] = \
                np.transpose(self.funSpace.shapeFunGradients[:, :, e]) @ self.funSpace.shapeFunGradients[:, :, e]

    def compGlobStiffStencil(self):
        # Compute stiffness stencil matrices K_e, such that K can be assembled via K = sum_e (lambda_e*K_e)
        # This can be done much more efficiently, but is precomputed and therefore not bottleneck.
        if self.locStiffGrad is None:
            self.compLocStiffGrad()

        eqInd, kIndex = self.compEquationIndices()

        globStiffStencil = np.empty((self.mesh.nEq**2, self.mesh.nCells))
        e = 0
        for cll in self.mesh.cells:
            gradLocK = np.zeros((4, 4, self.mesh.nEl))
            gradLocK[:, :, e] = self.locStiffGrad[e]
            gradLocK = gradLocK.flatten(order='F')
            Ke = sps.csr_matrix((gradLocK[kIndex], (eqInd[0], eqInd[1])))
            Ke_dense = sps.csr_matrix.todense(Ke)
            Ke = sps.csr_matrix(Ke_dense)
            self.globStiffGrad.append(Ke)
            globStiffStencil[:, e] = Ke_dense.flatten(order='F')
            e += 1
        globStiffStencil = sps.csr_matrix(globStiffStencil)
        globStiffStencil = PETSc.Mat().createAIJ(
            size=globStiffStencil.shape, csr=(globStiffStencil.indptr, globStiffStencil.indices, globStiffStencil.data))
        globStiffStencil.assemblyBegin()
        globStiffStencil.assemblyEnd()
        # indptr, colind, val = globStiffStencil.getValuesCSR()
        # print('PETSc to scipy = ', sps.csr_matrix((val, colind, indptr)))
        self.globStiffStencil = globStiffStencil

    def compSparsityPattern(self):
        # Computes sparsity pattern of stiffness matrix/stiffness matrix vector for fast matrix assembly
        testVec = PETSc.Vec().createSeq(self.mesh.nCells)
        testVec.setValues(range(self.mesh.nCells), np.ones(self.mesh.nCells))
        Kvec = PETSc.Vec().createSeq(self.mesh.nEq**2)

        self.globStiffStencil.mult(testVec, Kvec)
        self.vec_nonzero = np.nonzero(Kvec.array)
        self.nnz = len(self.vec_nonzero[0])     # important for memory allocation

        Ktmp = sps.csr_matrix(np.reshape(Kvec.array, (self.mesh.nEq, self.mesh.nEq), order='F'))
        self.indptr = Ktmp.indptr.copy()     # csr-like indices
        self.indices = Ktmp.indices.copy()

    def assemble(self, x):
        # x is numpy vector of permeability/conductivity
        # self.conductivity.setValues(self.rangeCells, x)
        self.globStiffStencil.mult(x, self.assemblyVector)
        self.matrix.setValuesCSR(self.indptr, self.indices, self.assemblyVector.getValues(self.vec_nonzero))
        self.matrix.assemblyBegin()
        self.matrix.assemblyEnd()


class RightHandSide:
    # This is the finite element force vector
    def __init__(self, mesh):
        self.vector = PETSc.Vec().createSeq(mesh.nEq)    # Force vector
        self.fluxBC = None
        self.sourceField = None
        self.naturalRHS = PETSc.Vec().createSeq(mesh.nEq)
        self.cellsWithEssentialBC = []    # precompute for performance
        self.findEssentialCells(mesh)
        # Use nnz = 0, PETSc will allocate  additional storage by itself
        self.rhsStencil = PETSc.Mat().createAIJ(size=(mesh.nEq, mesh.nCells), nnz=0)
        self.rhsStencil.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    def setFluxBC(self, mesh, flux):
        # Contribution due to flux boundary conditions

        self.fluxBC = np.zeros((4, mesh.nCells))
        for cll in mesh.cells:
            for edg in cll.edges:
                if edg.isNatural:
                    # Edge is a natural boundary
                    # This is only valid for the unit square domain!
                    # Find the edge outward normal vectors for generalization

                    # Short hand notation
                    ll = cll.vertices[0].coordinates    # lower left
                    ur = cll.vertices[2].coordinates    # upper right
                    if edg.vertices[0].coordinates[1] < np.finfo(float).eps and \
                       edg.vertices[1].coordinates[1] < np.finfo(float).eps:
                        # lower boundary
                        for i in range(4):
                            def fun(x):
                                q = flux(np.array([x, 0.0]))
                                q = - q[1]      # scalar product with boundary
                                if i == 0:
                                    N = (x - ur[0]) * (-ur[1])
                                elif i == 1:
                                    N = -(x - ll[0])*(- ur[1])
                                elif i == 2:
                                    N = (x - ll[0]) * (- ll[1])
                                elif i == 3:
                                    N = -(x - ur[0])*(- ll[1])
                                N /= cll.surface
                                return q*N

                            Intgrl = quad(fun, ll[0], ur[0])
                            self.fluxBC[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[0] > 1.0 - np.finfo(float).eps and \
                         edg.vertices[1].coordinates[0] > 1.0 - np.finfo(float).eps:
                        # right boundary
                        for i in range(4):
                            def fun(y):
                                q = flux(np.array([1.0, y]))
                                q = q[0]  # scalar product with boundary
                                # np.array([(xel[0] - upperRight[0]) * (xel[1] - upperRight[1]),
                                #           - (xel[0] - lowerLeft[0]) * (xel[1] - upperRight[1]),
                                #           (xel[0] - lowerLeft[0]) * (xel[1] - lowerLeft[1]),
                                #           - (xel[0] - upperRight[0]) * (xel[1] - lowerLeft[1])]) / Ael
                                if i == 0:
                                    N = (1.0 - ur[0]) * (y - ur[1])
                                elif i == 1:
                                    N = - (1.0 - ll[0]) * (y - ur[1])
                                elif i == 2:
                                    N = (1.0 - ll[0]) * (y - ll[1])
                                elif i == 3:
                                    N = - (1.0 - ur[0]) * (y - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[1], ur[1])
                            self.fluxBC[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[1] > 1.0 - np.finfo(float).eps and \
                         edg.vertices[1].coordinates[1] > 1.0 - np.finfo(float).eps:
                        # upper boundary
                        for i in range(4):
                            def fun(x):
                                q = flux(np.array([x, 1.0]))
                                q = q[1]  # scalar product with boundary
                                # np.array([(xel[0] - upperRight[0]) * (xel[1] - upperRight[1]),
                                #           - (xel[0] - lowerLeft[0]) * (xel[1] - upperRight[1]),
                                #           (xel[0] - lowerLeft[0]) * (xel[1] - lowerLeft[1]),
                                #           - (xel[0] - upperRight[0]) * (xel[1] - lowerLeft[1])]) / Ael
                                if i == 0:
                                    N = (x - ur[0]) * (1.0 - ur[1])
                                elif i == 1:
                                    N = - (x - ll[0]) * (1.0 - ur[1])
                                elif i == 2:
                                    N = (x - ll[0]) * (1.0 - ll[1])
                                elif i == 3:
                                    N = - (x - ur[0]) * (1.0 - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[0], ur[0])
                            self.fluxBC[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[0] < np.finfo(float).eps and \
                         edg.vertices[1].coordinates[0] < np.finfo(float).eps:
                        # left boundary
                        for i in range(4):
                            def fun(y):
                                q = flux(np.array([0.0, y]))
                                q = - q[0]  # scalar product with boundary
                                # np.array([(xel[0] - upperRight[0]) * (xel[1] - upperRight[1]),
                                #           - (xel[0] - lowerLeft[0]) * (xel[1] - upperRight[1]),
                                #           (xel[0] - lowerLeft[0]) * (xel[1] - lowerLeft[1]),
                                #           - (xel[0] - upperRight[0]) * (xel[1] - lowerLeft[1])]) / Ael
                                if i == 0:
                                    N = (- ur[0]) * (y - ur[1])
                                elif i == 1:
                                    N = ll[0] * (y - ur[1])
                                elif i == 2:
                                    N = ( - ll[0]) * (y - ll[1])
                                elif i == 3:
                                    N = ur[0] * (y - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[1], ur[1])
                            self.fluxBC[i, cll.number] += Intgrl[0]

    def setSourceField(self, mesh, sourceField):
        # Local force vector elements due to source field

        # Gauss points
        xi0 = -1.0/np.sqrt(3)
        xi1 = 1.0/np.sqrt(3)
        eta0 = -1.0/np.sqrt(3)
        eta1 = 1.0/np.sqrt(3)

        self.sourceField = np.zeros((4, mesh.nCells))

        e = 0
        for cll in mesh.cells:
            # Short hand notation
            x0 = cll.vertices[0].coordinates[0]
            x1 = cll.vertices[2].coordinates[0]
            y0 = cll.vertices[0].coordinates[1]
            y1 = cll.vertices[2].coordinates[1]

            # Coordinate transformation
            xI = .5*(x0 + x1) + .5*xi0*(x1 - x0)
            xII = .5*(x0 + x1) + .5*xi1*(x1 - x0)
            yI = .5*(y0 + y1) + .5*eta0*(y1 - y0)
            yII = .5*(y0 + y1) + .5*eta1*(y1 - y0)

            source = sourceField(cll.centroid)
            self.sourceField[0, e] = source*(1.0/cll.surface)*((xI - x1)*(yI - y1) + (xII - x1)*(yII - y1) +
                                                               (xI - x1)*(yII - y1) + (xII - x1)*(yI - y1))
            self.sourceField[1, e] = - source*(1.0/cll.surface)*((xI - x0)*(yI - y1) + (xII - x0) * (yII - y1) +
                                                                 (xI - x0)*(yII - y1) + (xII - x0)*(yI - y1))
            self.sourceField[2, e] = source*(1.0/cll.surface)*((xI - x0)*(yI - y0) + (xII - x0)*(yII - y0) +
                                                               (xI - x0)*(yII - y0) + (xII - x0) * (yI - y0))
            self.sourceField[3, e] = - source*(1.0/cll.surface)*((xI - x1)*(yI - y0) + (xII - x1)*(yII - y0) +
                                                                 (xI - x1)*(yII - y0) + (xII - x1)*(yI - y0))
            e += 1

    def setNaturalRHS(self, mesh, flux):
        # Sets the part of the RHS due to natural BC's and source field
        if self.fluxBC is None:
            self.setFluxBC(mesh, flux)

        naturalRHS = np.zeros(mesh.nEq)
        e = 0
        for cll in mesh.cells:
            v = 0
            for vtx in cll.vertices:
                if vtx.equationNumber is not None:
                    naturalRHS[vtx.equationNumber] += self.fluxBC[v, e]
                    if self.sourceField is not None:
                        naturalRHS[vtx.equationNumber] += self.sourceField[v, e]
                v += 1
            e += 1
        self.naturalRHS.setValues(range(mesh.nEq), naturalRHS)

    def findEssentialCells(self, mesh):
        for cll in mesh.cells:
            if cll.containsEssentialVertex:
                self.cellsWithEssentialBC.append(cll.number)

    def setRhsStencil(self, mesh, stiffnessMatrix):
        rhsStencil_np = np.zeros((mesh.nEq, mesh.nCells))
        for c in self.cellsWithEssentialBC:
            essBoundaryValues = np.zeros(4)
            i = 0
            for vtx in mesh.cells[c].vertices:
                if vtx.isEssential:
                    essBoundaryValues[i] = vtx.boundaryValue
                i += 1

            locEssBC = stiffnessMatrix.locStiffGrad[c] @ essBoundaryValues
            i = 0
            for vtx in mesh.cells[c].vertices:
                if vtx.equationNumber is not None:
                    rhsStencil_np[vtx.equationNumber, c] -= locEssBC[i]
                i += 1

        # Assemble PETSc matrix from numpy
        for c in self.cellsWithEssentialBC:
            for vtx in mesh.cells[c].vertices:
                if vtx.equationNumber is not None:
                    self.rhsStencil.setValue(vtx.equationNumber, c, rhsStencil_np[vtx.equationNumber, c])
        self.rhsStencil.assemblyBegin()
        self.rhsStencil.assemblyEnd()

    def assemble(self, x):
        # x is a PETSc vector of conductivity/permeability
        self.rhsStencil.multAdd(x, self.naturalRHS, self.vector)

