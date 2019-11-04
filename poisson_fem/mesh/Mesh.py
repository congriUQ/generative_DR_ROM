'''FEM Mesh base class'''
from matplotlib import pyplot as plt
import numpy as np


class Mesh:
    def __init__(self):
        self._vertices = []
        self._edges = []
        self.__cells = []
        self._nVertices = 0
        self._nEdges = 0
        self.nCells = 0

    def createVertex(self, coordinates):
        # Creates a vertex and appends it to vertex list
        vtx = Vertex(coordinates)
        self._vertices.append(vtx)
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
        self.__cells.append(cll)

        # Link cell to vertices
        for vtx in vertices:
            vtx.addCell(cll)

        # Link cell to edges
        for edg in edges:
            edg.addCell(cll)

        self.nCells += 1

    def plot(self):
        n = 0
        for vtx in self._vertices:
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
        for cll in self.__cells:
            if cll is not None:
                plt.text(cll.centroid[0], cll.centroid[1], str(n))
            n += 1

        plt.xlabel('x')
        plt.ylabel('y')


class RectangularMesh(Mesh):
    def __init__(self, gridX, gridY=None):
        # Grid vectors gridX, gridY give the edge lengths

        super().__init__()
        if gridY is None:
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
                if self._vertices[x + y*nx].coordinates[0] > 0:
                    self.createEdge(self._vertices[x + y*nx - 1], self._vertices[x + y*nx])
        for y in range(ny):
            for x in range(nx):
                if self._vertices[x + y*nx].coordinates[1] > 0:
                    self.createEdge(self._vertices[x + (y - 1)*nx], self._vertices[x + y*nx])

        # Create cells
        nx -= 1
        ny -= 1
        n = 0   # cell index
        for y in range(ny):
            for x in range(nx):
                vtx = [self._vertices[x + y*(nx + 1)], self._vertices[x + y*(nx + 1) + 1],
                       self._vertices[x + (y + 1)*(nx + 1) + 1], self._vertices[x + (y + 1)*(nx + 1)]]
                edg = [self._edges[n], self._edges[nx*(ny + 1) + n + y + 1], self._edges[n + nx],
                       self._edges[nx*(ny + 1) + n + y]]
                self.createCell(vtx, edg)
                n += 1


class Cell:
    def __init__(self, vertices, edges):
        # Vertices and edges must be sorted according to local vertex/edge number!
        self._vertices = vertices
        self._edges = edges
        self.centroid = []
        self.computeCentroid()

    def computeCentroid(self):
        # Compute cell centroid
        self.centroid = np.zeros(2)
        for vtx in self._vertices:
            self.centroid += vtx.coordinates
        self.centroid /= len(self._vertices)

    def deleteEdges(self, indices):
        # Deletes edges according to indices by setting them to None
        for i in indices:
            self._edges[i] = None

    def inside(self, x):
        # Checks if point x is inside of cell
        return (self._vertices[0].coordinates[0] - np.finfo(float).eps < x[0]
                <= self._vertices[2].coordinates[0] + np.finfo(float).eps and
                self._vertices[0].coordinates[1] - np.finfo(float).eps < x[1]
                <= self._vertices[2].coordinates[1] + np.finfo(float).eps)


class Edge:
    def __init__(self, vtx0, vtx1):
        self._vertices = [vtx0, vtx1]
        self.__cells = []
        self.length = np.linalg.norm(vtx0.coordinates - vtx1.coordinates)

    def addCell(self, cell):
        self.__cells.append(cell)

    def plot(self, n):
        plt.plot([self._vertices[0].coordinates[0], self._vertices[1].coordinates[0]],
                 [self._vertices[0].coordinates[1], self._vertices[1].coordinates[1]], linewidth=.5, color='r')
        plt.text(.5*(self._vertices[0].coordinates[0] + self._vertices[1].coordinates[0]),
                 .5*(self._vertices[0].coordinates[1] + self._vertices[1].coordinates[1]), str(n), color='r')


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