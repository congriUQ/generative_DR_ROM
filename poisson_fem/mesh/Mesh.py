'''FEM Mesh base class'''
from matplotlib import pyplot as plt
import numpy as np


class Mesh:
    def __init__(self):
        self.vertices = []
        self._edges = []
        self.cells = []
        self._nVertices = 0
        self._nEdges = 0
        self.nCells = 0

    def createVertex(self, coordinates):
        # Creates a vertex and appends it to vertex list
        vtx = Vertex(coordinates)
        self.vertices.append(vtx)
        self._nVertices += 1

    def createEdge(self, vtx0, vtx1):
        # Creates an edge between vertices specified by vertex indices and adds it to edge list
        edg = Edge(vtx0, vtx1)
        self._edges.append(edg)

        # Link edges to vertices
        vtx0.addEdge(edg)
        vtx1.addEdge(edg)

        self._nEdges += 1

    def createCell(self, vertices, edges):
        # Creates a cell and appends it to cell list
        cll = Cell(vertices, edges)
        self.cells.append(cll)

        # Link cell to vertices
        for vtx in vertices:
            vtx.addCell(cll)

        # Link cell to edges
        for edg in edges:
            edg.addCell(cll)

        self.nCells += 1

    def plot(self):
        n = 0
        for vtx in self.vertices:
            if vtx is not None:
                vtx.plot(n)
                n += 1

        n = 0
        for edg in self._edges:
            if edg is not None:
                edg.plot(n)
                n += 1

        # plot cell number into cell
        n = 0
        for cll in self.cells:
            if cll is not None:
                plt.text(cll.centroid[0], cll.centroid[1], str(n))
            n += 1

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
        for y in yCoord:
            for x in xCoord:
                self.createVertex(np.array([x, y]))

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
                edg = [self._edges[n], self._edges[nx*(ny + 1) + n + y + 1], self._edges[n + nx],
                       self._edges[nx*(ny + 1) + n + y]]
                self.createCell(vtx, edg)
                n += 1

        # minor stuff
        self.nElX = len(gridX)
        self.nElY = len(gridY)
        self.nEl = self.nElX * self.nElY
        self.shapeFunGradients = None
        self.locStiffGrad = None

    def compShapeFunGradMat(self):
        # Computes shape function gradient matrices B, see Fish & Belytshko

        # Gauss quadrature points
        xi0 = -1.0/np.sqrt(3)
        xi1 = 1.0/np.sqrt(3)

        self.shapeFunGradients = np.empty((8, 4, self.nEl))
        for e in range(self.nEl):
            # short hand notation
            x0 = self.cells[e].vertices[0].coordinates[0]
            x1 = self.cells[e].vertices[1].coordinates[0]
            y0 = self.cells[e].vertices[0].coordinates[1]
            y3 = self.cells[e].vertices[3].coordinates[1]

            # coordinate transformation
            xI = .5*(x0 + x1) + .5*xi0*(x1 - x0)
            xII = .5*(x0 + x1) + .5*xi1*(x1 - x0)
            yI = .5*(y0 + y3) + .5*xi0*(y3 - y0)
            yII = .5*(y0 + y3) + .5*xi1*(y3 - y0)

            # B matrices for bilinear shape functions
            B0 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xI - x1, x0 - xI, xI - x0, x1 - xI]])
            B1 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B2 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B3 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xI - x1, x0 - xI, xI - x0, x1 - xI]])

            # Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence we take the
            # additional factor of sqrt(A)/2 onto B
            self.shapeFunGradients[:, :, e] = (1/(2*np.sqrt(self.cells[e].surface)))*np.concatenate((B0, B1, B2, B3))

    def compLocStiffGrad(self):
        # Compute local stiffness matrix gradients w.r.t. diffusivities

        if self.shapeFunGradients is None:
            self.compShapeFunGradMat()

        self.locStiffGrad = np.empty((4, 4, self.nEl))
        for e in range(self.nEl):
            self.locStiffGrad[:, :, e] = \
                np.transpose(self.shapeFunGradients[:, :, e]) @ self.shapeFunGradients[:, :, e]

    def compGlobStiffStencil(self):
        # tbd
        pass


class Cell:
    def __init__(self, vertices, edges):
        # Vertices and edges must be sorted according to local vertex/edge number!
        self.vertices = vertices
        self._edges = edges
        self.centroid = []
        self.computeCentroid()
        self.surface = self._edges[0].length*self._edges[1].length

    def computeCentroid(self):
        # Compute cell centroid
        self.centroid = np.zeros(2)
        for vtx in self.vertices:
            self.centroid += vtx.coordinates
        self.centroid /= len(self.vertices)

    def deleteEdges(self, indices):
        # Deletes edges according to indices by setting them to None
        for i in indices:
            self._edges[i] = None

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

    def addCell(self, cell):
        self.cells.append(cell)

    def plot(self, n):
        plt.plot([self.vertices[0].coordinates[0], self.vertices[1].coordinates[0]],
                 [self.vertices[0].coordinates[1], self.vertices[1].coordinates[1]], linewidth=.5, color='r')
        plt.text(.5*(self.vertices[0].coordinates[0] + self.vertices[1].coordinates[0]),
                 .5*(self.vertices[0].coordinates[1] + self.vertices[1].coordinates[1]), str(n), color='r')


class Vertex:
    def __init__(self, coordinates=np.zeros((2, 1))):
        self.coordinates = coordinates
        self.cells = []
        self.edges = []

    def addCell(self, cell):
        self.cells.append(cell)

    def addEdge(self, edge):
        self.edges.append(edge)

    def plot(self, n):
        p = plt.plot(self.coordinates[0], self.coordinates[1], 'bx', linewidth=2.0, markersize=8.0)
        plt.text(self.coordinates[0], self.coordinates[1], str(n), color='b')