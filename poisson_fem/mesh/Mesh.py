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
        #Creates an edge between vertices specified by vertex indices and adds it to edge list
        edg = Edge(vtx0, vtx1)
        self.__edges.append(edg)

        # Link edges to vertices
        vtx0.addEdge(edg)
        vtx1.addEdge(edg)


class Vertex():
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


class Edge():
    def __init__(self, vtx0, vtx1):
        self.__vertices = [vtx0, vtx1]
        self.__cells = []
        self.length = np.linalg.norm(vtx0.coordinates - vtx1.coordinates)


    def addCell(self, cell):
        self.__cells.append(cell)


    def plot(self):
        plt.plot([self.__vertices[0].coordinates[0], self.__vertices[1].coordinates[0]],
                 [self.__vertices[0].coordinates[1], self.__vertices[1].coordinates[1]], linewidth=.5, color='k')