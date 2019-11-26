import socket
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

'''File for Stokes and Darcy training/test data'''


class Data:
    # Base class for Stokes/Darcy training/testing data
    def __init__(self, samples_in, dtype=torch.float):
        self.dtype = dtype
        self.resolution = 256               # mesh resolution
        self.samples_in = samples_in              # list of sample numbers
        self.n_samples_in = len(samples_in)
        self.n_samples_out = 16             # change this to number of output samples!!
        self.path = ''                    # path to data folder
        self.output = []                  # response field, i.e., output data
        self.input = []


class StokesData(Data):
    def __init__(self, samples_in):
        super().__init__(samples_in)
        # Number of exclusions
        self.n_excl_dist = 'logn'                         # Number of exclusions distribution
        self.n_excl_dist_params = [6.8, .2]
        self.excl_margins = [.003, .003, .003, .003]     # closest distance of exclusion to boundary

        # Radii distribution
        self.r_dist = 'lognGP'
        self.r_dist_params = [-4.1, .7]
        self.r_GP_sigma = .01
        self.r_GP_length = .05

        # Center coordinate distribution
        self.coord_dist = 'GP'
        self.coord_dist_cov = 'squaredExponential'

        # Density of exclusions
        self.dens_length_scale = .08
        self.dens_sigm_scale = 1.2

        # Boundary conditions
        self.boundary_conditions = [.0, 1.0, 1.0, -0.0]

        # Mesh data
        self.vertex_coordinates = []
        self.cell_connectivity = []
        self.P = []                                         # Pressure response at vertices
        self.V = []                                         # Velocity at vertices
        self.X = []
        self.microstructure_excl_radii = []
        self.microstructure_excl_centers = []

        self.imgX = []                                      # Microstructure pixel coordinates
        self.img_resolution = 256
        # Initialization -- dtype will change to self.dtype
        self.microstructure_image = torch.zeros(self.n_samples_in, self.img_resolution**2, dtype=torch.bool)

    def set_path_name(self):
        assert len(self.path) == 0, 'Data path already set'
        self.path += '/home/constantin/'
        
        if socket.gethostname() == 'workstation1-room0436':
            # office workstation
            self.path += 'cluster/'
            
        self.path += 'python/data/stokesEquation/meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
                     str(self.excl_margins[0]) + '_' + str(self.excl_margins[1]) + '_' + str(self.excl_margins[2]) +\
                     '_' + str(self.excl_margins[3]) + '/N~' + self.n_excl_dist + '/mu=' + \
                     str(self.n_excl_dist_params[0]) + '/sigma=' + str(self.n_excl_dist_params[1]) + \
                     '/x~' + self.coord_dist

        if self.coord_dist == 'GP':
            self.path += '/cov=' + self.coord_dist_cov + '/l=' + str(self.dens_length_scale) + '/sig_scale=' + \
                         str(self.dens_sigm_scale) + '/r~' + self.r_dist
        elif self.coord_dist == 'engineered' or self.coord_dist == 'tiles':
            self.path += '/r~' + self.r_dist
        else:
            self.path += '/mu=.5_.5' + '/cov=' + self.coord_dist_cov + '/r~' + self.r_dist

        if self.r_dist == 'lognGP':
            self.path += '/mu=' + str(self.r_dist_params[0]) + '/sigma=' + str(self.r_dist_params[1]) + '/sigmaGP_r=' +\
                            str(self.r_GP_sigma) + '/l=' + str(self.r_GP_length) + '/'
        else:
            self.path += '/mu=' + str(self.r_dist_params[0]) + '/sigma=' + str(self.r_dist_params[1]) + '/'

    def read_data(self, quantities):
        if len(self.path) == 0:
            self.set_path_name()
        folderName = self.path + 'p_bc=0.0/' + 'u_x=' + str(self.boundary_conditions[1]) + \
                         str(self.boundary_conditions[3]) + 'x[1]_u_y=' + str(self.boundary_conditions[2]) + \
                         str(self.boundary_conditions[3]) + 'x[0]'

        sol_quantities = ['P', 'V', 'X']
        for i, n in enumerate(self.samples_in):
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
                except FileNotFoundError:
                    microstructureFileName = self.path + 'microstructureInformation_nomesh' + str(n) + '.mat'
                    microstructFile = sio.loadmat(microstructureFileName)
                self.microstructure_excl_radii.append(microstructFile['diskRadii'].flatten())
                self.microstructure_excl_centers.append(microstructFile['diskCenters'])
            if 'IMG' in quantities:
                try:
                    self.microstructure_image[i] = torch.tensor(np.load((self.path + 'microstructure_image' +
                                                        str(n) + '_res=' + str(self.img_resolution)) + '.npy'),
                                                          dtype=torch.bool)
                except FileNotFoundError:
                    # If not transformed to image yet --
                    # should only happen for the first sample
                    self.input2img(save=True)
                    self.microstructure_image[i] = torch.tensor(np.load(self.path + 'microstructure_image' +
                                                          str(n) + '_res=' + str(self.img_resolution) + '.npy'),
                                                          dtype=torch.bool)
        if 'IMG' in quantities:
            # Change data type from bool to dtype
            self.microstructure_image = self.microstructure_image.type(self.dtype)
            # compute pixel coordinates
            self.get_pixel_coordinates()

    def get_pixel_coordinates(self):
        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])
        self.imgX = torch.cat([xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)], 1)

    def input2img(self, save=False):
        if len(self.microstructure_excl_radii) == 0:
            self.read_data(['M'])

        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])

        # loop over exclusions
        for i, n in enumerate(self.samples_in):
            r2 = self.microstructure_excl_radii[i]**2.0

            for nEx in range(len(self.microstructure_excl_radii[i])):
                tmp = ((xx - self.microstructure_excl_centers[i][nEx, 0])**2.0 +
                       (yy - self.microstructure_excl_centers[i][nEx, 1])**2.0 <= r2[nEx])
                tmp = tmp.flatten()
                self.microstructure_image[i] = self.microstructure_image[i] | tmp
            # self.microstructure_image[-1] = self.microstructure_image[-1].type(self.dtype)
            if save:
                # save to pytorch tensor
                np.save(self.path + 'microstructure_image' + str(n) +
                           '_res=' + str(self.img_resolution), self.microstructure_image[i].detach().numpy())

        self.microstructure_image = self.microstructure_image.type(self.dtype)
        # CrossEntropyLoss wants dtype long == int64
        # self.microstructure_image = self.microstructure_image.type(torch.long)

    def reshape_microstructure_image(self):
        # Reshapes flattened input data to 2D image of img_resolution x img_resolution
        # Should be a tensor ox shape (batchsize x channels x nPixelsH x nPixelsW)
        tmp = torch.zeros(self.n_samples_in, 1, self.img_resolution, self.img_resolution, dtype=self.dtype)
        for n in range(self.n_samples_in):
            tmp[n, 0, :, :] = torch.reshape(self.microstructure_image[n], (self.img_resolution, self.img_resolution))
        self.microstructure_image = tmp

    def plot_microstruct(self, sampleNumber):
        if len(self.microstructure_image) == 0:
            self.input2img()
        plt.imshow(torch.reshape(self.microstructure_image[sampleNumber],
                                 (self.img_resolution, self.img_resolution)), cmap='binary')

