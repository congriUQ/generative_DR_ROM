'''FEM Mesh base class'''
from matplotlib import pyplot as plt
import numpy as np

class Mesh():
    def __init__(self, vertices=[], edges=[], cells=[]):
        self.__vertices = vertices
        self.__edges = edges
        self.__cells = cells
        self._nVertices = 0
        self._nEdges = 0
        self.nCells = 0

    def createVertex(self, coordinates):
        # Creates a vertex and appends it to vertex list
        vtx = Vertex(coordinates)
        self.__vertices.append(vtx)
        self._nVertices += 1

    def createEdge(self, vtx0, vtx1):
        # Creates an edge between vertices specified by vertex indices and adds it to edge list
        edg = Edge(vtx0, vtx1)
        self.__edges.append(edg)

        # Link edges to vertices
        vtx0.addEdge(edg)
        vtx1.addEdge(edg)

        self.nEdges += 1

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
        for vtx in self.__vertices:
            if vtx is not None:
                vtx.plot()

        for edg in self.__edges:
            if edg is not None:
                edg.plot()

        # plot cell number into cell
        n = 0
        for cll in self.__cells:
            if cll is not None:
                plt.text(cll.__centroid[0], cll.__centroid[1], str(n))
            n += 1


class Cell:
    def __init__(self, vertices, edges):
        # Vertices and edges must be sorted according to local vertex/edge number!
        self.__vertices = vertices
        self.__edges = edges
        self.__centroid = []
        self.computeCentroid()

    def computeCentroid(self):
        # Compute cell centroid
        self.__centroid = np.zeros(2)
        for vtx in self.__vertices:
            self.__centroid += vtx.coordinates
        self.__centroid /= len(self.__vertices)

    def deleteEdges(self, indices):
        # Deletes edges according to indices by setting them to None
        for i in indices:
            self.__edges[i] = None

    def inside(self, x):
        # Checks if point x is inside of cell
        return (self.__vertices[0].coordinates[0] - np.finfo(float).eps < x[0]
                <= self.__vertices[2].coordinates[0] + np.finfo(float).eps and
                self.__vertices[0].coordinates[1] - np.finfo(float).eps < x[1]
                <= self.__vertices[2].coordinates[1] + np.finfo(float).eps)

class Edge:
    def __init__(self, vtx0, vtx1):
        self.__vertices = [vtx0, vtx1]
        self.__cells = []
        self.length = np.linalg.norm(vtx0.coordinates - vtx1.coordinates)

    def addCell(self, cell):
        self.__cells.append(cell)

    def plot(self):
        plt.plot([self.__vertices[0].coordinates[0], self.__vertices[1].coordinates[0]],
                 [self.__vertices[0].coordinates[1], self.__vertices[1].coordinates[1]], linewidth=.5, color='k')


class Vertex:
    def __init__(self, coordinates=np.zeros((2, 1))):
        self.coordinates = coordinates
        self.cells = []
        self.edges = []

    def addCell(self, cell):
        self.cells.append(cell)

    def addEdge(self, edge):
        self.edges.append(edge)

    def plot(self):
        p = plt.plot(self.coordinates[0], self.coordinates[1], 'bx', linewidth=2.0, markersize=8.0)