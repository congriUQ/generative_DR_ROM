import socket
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

'''File for Stokes and Darcy training/test data'''


class Data:
    # Base class for Stokes/Darcy training/testing data
    def __init__(self, samples):
        self.resolution = 256               # mesh resolution
        self.samples = samples              # list of sample numbers
        self.nSamples = len(samples)
        self.path = ''                    # path to data folder
        self.output = []                  # response field, i.e., output data
        self.input = []


class StokesData(Data):
    def __init__(self, samples):
        super().__init__(samples)
        # Number of exclusions
        self.nExclDist = 'logn'                         # Number of exclusions distribution
        self.nExclDistParams = [7.8, .2]
        self.exclMargins = [.003, .003, .003, .003]     # closest distance of exclusion to boundary

        # Radii distribution
        self.rDist = 'lognGP'
        self.rDistParams = [-5.23, .3]
        self.rGPsigma = .4
        self.rGPlength = .05

        # Center coordinate distribution
        self.coordDist = 'GP'
        self.coordDistCov = 'squaredExponential'

        # Density of exclusions
        self.densLengthScale = .08
        self.densSigmScale = 1.2

        # Boundary conditions
        self.boundaryConditions = [.0, 1.0, 1.0, -0.0]

        # Mesh data
        self.vertexCoordinates = []
        self.cellConnectivity = []
        self.P = []                                         # Pressure response at vertices
        self.V = []                                         # Velocity at vertices
        self.X = []
        self.microstructR = []
        self.microstructX = []
        self.microstructBitmap = []
        self.bitmapResolution = 256

    def setPathName(self):
        assert len(self.path) == 0, 'Data path already set'
        if socket.gethostname() == 'workstation1-room0436':
            # office workstation
            self.path += '/home/constantin/cluster/python/data/stokesEquation/'

        if self.coordDist == 'GP':
            self.path += 'meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
            str(self.exclMargins[0]) + '_' + str(self.exclMargins[1]) + '_' + str(self.exclMargins[2]) + '_' + \
            str(self.exclMargins[3]) + '/N~' + self.nExclDist + '/mu=' + str(self.nExclDistParams[0]) + '/sigma=' + \
            str(self.nExclDistParams[1]) + '/x~' + self.coordDist + '/cov=' + self.coordDistCov + '/l=' + \
            str(self.densLengthScale) + '/sig_scale=' + str(self.densSigmScale) + '/r~' + self.rDist
        elif self.coordDist == 'engineered':
            self.path += 'meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
            str(self.exclMargins[0]) + '_' + str(self.exclMargins[1]) + '_' + str(self.exclMargins[3]) + '_' + \
            str(self.exclMargins[3]) + '/N~' + self.nExclDist + '/mu=' + str(self.nExclDistParams[0]) + '/sigma=' + \
            str(self.nExclDistParams[1]) + '/x~' + self.coordDist + '/r~' + self.rDist
        elif self.coordDist == 'tiles':
            self.path += 'meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
            str(self.exclMargins[0]) + '_' + str(self.exclMargins[1]) + '_' + str(self.exclMargins[2]) + '_' + \
            str(self.exclMargins[3]) + '/N~' + self.nExclDist + '/mu=' + str(self.nExclDistParams[0]) + '/sigma=' + \
            str(self.nExclDistParams[1]) + '/x~' + self.coordDist + '/r~' + self.rDist
        else:
            self.path += 'meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
            str(self.exclMargins[0]) + '_' + str(self.exclMargins[1]) + '_' + str(self.exclMargins[2]) + '_' + \
            str(self.exclMargins[3]) + '/N~' + self.nExclDist + '/mu=' + str(self.nExclDistParams[0]) + '/sigma=' + \
            str(self.nExclDistParams[1]) + '/x~' + self.coordDist + '/mu=.5_.5' + '/cov=' + self.coordDistCov + \
            '/r~' + self.rDist

        if self.rDist == 'lognGP':
            self.path +=  '/mu=' + str(self.rDistParams[0]) + '/sigma=' + str(self.rDistParams[1]) + '/sigmaGP_r=' +\
            str(self.rGPsigma) + '/l=' + str(self.rGPlength) + '/'
        else:
            self.path += '/mu=' + str(self.rDistParams[0]) + '/sigma=' + str(self.rDistParams[1]) + '/'

    def readData(self, quantities):
        if len(self.path) == 0:
            self.setPathName()
        folderName = self.path + 'p_bc=0.0/' + 'u_x=' + str(self.boundaryConditions[1]) + \
                         str(self.boundaryConditions[3]) + 'x[1]_u_y=' + str(self.boundaryConditions[2]) + \
                         str(self.boundaryConditions[3]) + 'x[0]'

        for n in self.samples:
            solutionFileName = folderName + '/solution' + str(n) + '.mat'
            file = sio.loadmat(solutionFileName)
            if 'P' in quantities:
                self.P.append(file['p'])
            if 'V' in quantities:
                self.V.append(file['u'])
            if 'X' in quantities:
                self.X.append(file['x'])
            if 'M' in quantities:
                microstructureFileName = self.path + 'microstructureInformation' + str(n) + '.mat'
                microstructFile = sio.loadmat(microstructureFileName)
                self.microstructR.append(microstructFile['diskRadii'].flatten())
                self.microstructX.append(microstructFile['diskCenters'])

    def input2bitmap(self):
        if len(self.microstructR) == 0:
            self.readData(['M'])

        xx, yy = np.meshgrid(np.linspace(0, 1, self.bitmapResolution), np.linspace(0, 1, self.bitmapResolution))
        # loop over exclusions
        i = 0
        for n in self.samples:
            r2 = self.microstructR[i]**2.0
            self.microstructBitmap.append(np.zeros((self.bitmapResolution, self.bitmapResolution), dtype=bool))

            for nEx in range(len(self.microstructR[i])):
                self.microstructBitmap[-1] = np.logical_or(self.microstructBitmap[-1],
                ((xx - self.microstructX[i][nEx, 0])**2.0 + (yy - self.microstructX[i][nEx, 1])**2.0 <= r2[nEx]))
            i += 1

    def plotMicrostruct(self, sampleNumber):
        if len(self.microstructBitmap) == 0:
            self.input2bitmap()
        plt.imshow(self.microstructBitmap[sampleNumber], cmap='binary')

