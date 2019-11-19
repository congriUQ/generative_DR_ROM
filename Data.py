import socket
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

'''File for Stokes and Darcy training/test data'''


class Data:
    # Base class for Stokes/Darcy training/testing data
    def __init__(self, samples, dtype=torch.float):
        self.dtype = dtype
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
        self.nExclDistParams = [6.8, .2]
        self.exclMargins = [.003, .003, .003, .003]     # closest distance of exclusion to boundary

        # Radii distribution
        self.rDist = 'lognGP'
        self.rDistParams = [-4.1, .7]
        self.rGPsigma = .01
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

        self.imgX = []                                      # Microstructure pixel coordinates
        self.imgResolution = 256
        # Initialization -- dtype will change to self.dtype
        self.microstructImg = torch.zeros(self.nSamples, self.imgResolution**2, dtype=torch.bool)

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

        i = 0
        sol_quantities = ['P', 'V', 'X']
        for n in self.samples:
            if any(qnt in quantities for qnt in sol_quantities):
                # to avoid loading overhead
                solutionFileName = folderName + '/solution' + str(n) + '.mat'
                file = sio.loadmat(solutionFileName)
            if 'P' in quantities:
                self.P.append(file['p'])
            if 'V' in quantities:
                self.V.append(file['u'])
            if 'X' in quantities:
                self.X.append(file['x'])
            if 'M' in quantities:
                try:
                    microstructureFileName = self.path + 'microstructureInformation' + str(n) + '.mat'
                    microstructFile = sio.loadmat(microstructureFileName)
                except:
                    microstructureFileName = self.path + 'microstructureInformation_nomesh' + str(n) + '.mat'
                    microstructFile = sio.loadmat(microstructureFileName)
                self.microstructR.append(microstructFile['diskRadii'].flatten())
                self.microstructX.append(microstructFile['diskCenters'])
            if 'IMG' in quantities:
                try:
                    self.microstructImg[i] = torch.tensor(np.load((self.path + 'microstructImg' +
                                                        str(n) + '_res=' + str(self.imgResolution)) + '.npy'),
                                                          dtype=torch.bool)
                except:
                    # If not transformed to image yet --
                    # should only happen for the first sample
                    self.input2img(save=True)
                    self.microstructImg[i] = torch.tensor(np.load(self.path + 'microstructImg' +
                                                          str(n) + '_res=' + str(self.imgResolution) + '.npy'),
                                                          dtype=torch.bool)
            i += 1
        if 'IMG' in quantities:
            # Change data type from bool to dtype
            self.microstructImg = self.microstructImg.type(self.dtype)
            # compute pixel coordinates
            self.getImgX()

    def getImgX(self):
        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.imgResolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.imgResolution, dtype=self.dtype)])
        self.imgX = torch.cat([xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)], 1)

    def input2img(self, save=False):
        if len(self.microstructR) == 0:
            self.readData(['M'])

        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.imgResolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.imgResolution, dtype=self.dtype)])

        # loop over exclusions
        i = 0
        for n in self.samples:
            r2 = self.microstructR[i]**2.0

            for nEx in range(len(self.microstructR[i])):
                tmp = ((xx - self.microstructX[i][nEx, 0])**2.0 + (yy - self.microstructX[i][nEx, 1])**2.0 <= r2[nEx])
                tmp = tmp.flatten()
                self.microstructImg[i] = self.microstructImg[i] | tmp
            # self.microstructImg[-1] = self.microstructImg[-1].type(self.dtype)
            if save:
                # save to pytorch tensor
                np.save(self.path + 'microstructImg' + str(n) +
                           '_res=' + str(self.imgResolution), self.microstructImg[i].detach().numpy())
            i += 1
        self.microstructImg = self.microstructImg.type(self.dtype)
        # CrossEntropyLoss wants dtype long == int64
        # self.microstructImg = self.microstructImg.type(torch.long)

    def plotMicrostruct(self, sampleNumber):
        if len(self.microstructImg) == 0:
            self.input2img()
        plt.imshow(torch.reshape(self.microstructImg[sampleNumber],
                                 (self.imgResolution, self.imgResolution)), cmap='binary')

