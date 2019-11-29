import socket
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

'''File for Stokes and Darcy training/test data'''


class Data:
    # Base class for Stokes/Darcy training/testing data
    def __init__(self, supervised_samples, unsupervised_samples, dtype=torch.float):
        assert len(supervised_samples.intersection(unsupervised_samples)) == 0, "samples both supervised/" \
                                                                                "unsupervised"
        self.dtype = dtype
        self.resolution = 256               # mesh resolution
        self.supervised_samples = supervised_samples              # list of sample numbers
        self.n_supervised_samples = len(supervised_samples)
        self.unsupervised_samples = unsupervised_samples
        self.n_unsupervised_samples = len(unsupervised_samples)
        self.path = ''                    # path to input data folder
        self.solution_folder = ''
        self.output = []                  # response field, i.e., output data
        self.input = []

        self.output_resolution = 129    # resolution of output image interpolation


class StokesData(Data):
    def __init__(self, supervised_samples=set(), unsupervised_samples=set()):
        super().__init__(supervised_samples, unsupervised_samples)
        # Number of exclusions
        self.n_excl_dist = 'logn'                         # Number of exclusions distribution
        self.n_excl_dist_params = [7.8, .2]
        self.excl_margins = [.003, .003, .003, .003]     # closest distance of exclusion to boundary

        # Radii distribution
        self.r_dist = 'lognGP'
        self.r_dist_params = [-5.23, .3]
        self.r_GP_sigma = .4
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
        self.P = torch.zeros(self.n_supervised_samples, self.output_resolution**2)     # Pressure response at vertices
        self.V = []                                         # Velocity at vertices
        self.X = []
        self.microstructure_excl_radii = []
        self.microstructure_excl_centers = []

        self.imgX = []                                      # Microstructure pixel coordinates
        self.img_resolution = 256
        # Initialization -- dtype will change to self.dtype
        # Convention: supervised samples first, then unsupervised samples
        self.microstructure_image = torch.zeros(self.n_supervised_samples + self.n_unsupervised_samples,
                                                self.img_resolution**2, dtype=torch.bool)

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

        self.solution_folder = self.path + 'p_bc=0.0/' + 'u_x=' + str(self.boundary_conditions[1]) + \
                          str(self.boundary_conditions[3]) + 'x[1]_u_y=' + str(self.boundary_conditions[2]) + \
                          str(self.boundary_conditions[3]) + 'x[0]'

    def read_data(self):
        if len(self.path) == 0 or len(self.solution_folder) == 0:
            # set path to file if not yet set
            self.set_path_name()

        for i, n in enumerate(self.supervised_samples):
            solutionFileName = self.solution_folder + '/solution' + str(n) + '.mat'
            file = sio.loadmat(solutionFileName)
            # self.P.append(file['p'])
            # self.X.append(file['x'])
            p = file['p']
            x = file['x']
            p_interp = self.interpolate_pressure(x, p, n)
            self.P[i, :] = torch.tensor(p_interp.flatten())
        
        for i, n in enumerate(self.supervised_samples | self.unsupervised_samples):
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

        self.microstructure_image = self.microstructure_image.type(self.dtype)
        # self.get_pixel_coordinates()

    def interpolate_pressure(self, x, p, n, shift=True):
        # shift shifts the data s.t. p(x = 0) = 0
        file_name = self.solution_folder + '/p_interp' + str(n) + '_res=' + str(self.output_resolution) + '.mat'
        file = sio.loadmat(file_name)
        p_interp = file['p_interp']
        if shift:
            p_interp -= p_interp[0]
        # if os.path.isfile(file_name):
        #
        #     return np.load(file_name)
        # else:
        #     # interpolant = interp.interp2d(x[:, 0], x[:, 1], p.flatten())
        #     print('before interp')
        #     sys.stdout.flush()
        #     interpolant = interp.SmoothBivariateSpline(x[:, 0], x[:, 1], p.flatten())
        #     print('before mesh')
        #     sys.stdout.flush()
        #     xx, yy = np.meshgrid(np.linspace(0, 1, self.output_resolution), np.linspace(0, 1, self.output_resolution))
        #     print('before eval')
        #     sys.stdout.flush()
        #     p_interp = interpolant(xx.flatten(), yy.flatten())
        #     np.save(self.solution_folder + '/interpolated_pressure' + str(n) + '_res=' + str(self.output_resolution),
        #             p_interp)
        return p_interp

    def read_microstructure_information(self):
        if len(self.path) == 0:
            # set path to file if not yet set
            self.set_path_name()

        for n in (self.supervised_samples | self.unsupervised_samples):
            try:
                microstructureFileName = self.path + 'microstructureInformation' + str(n) + '.mat'
                microstructFile = sio.loadmat(microstructureFileName)
            except FileNotFoundError:
                microstructureFileName = self.path + 'microstructureInformation_nomesh' + str(n) + '.mat'
                microstructFile = sio.loadmat(microstructureFileName)
            self.microstructure_excl_radii.append(microstructFile['diskRadii'].flatten())
            self.microstructure_excl_centers.append(microstructFile['diskCenters'])

    def get_pixel_coordinates(self):
        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])
        self.imgX = torch.cat([xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)], 1)

    def input2img(self, save=False):
        if len(self.microstructure_excl_radii) == 0:
            self.read_microstructure_information()

        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])

        # loop over exclusions
        for i, n in enumerate(self.supervised_samples | self.unsupervised_samples):
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
        tmp = torch.zeros(self.n_supervised_samples + self.n_unsupervised_samples, 1,
                          self.img_resolution, self.img_resolution, dtype=self.dtype)
        for i in range(self.n_supervised_samples + self.n_unsupervised_samples):
            tmp[i, 0, :, :] = torch.reshape(self.microstructure_image[i], (self.img_resolution, self.img_resolution))
        self.microstructure_image = tmp

    def plot_microstruct(self, sampleNumber):
        if len(self.microstructure_image) == 0:
            self.input2img()
        plt.imshow(torch.reshape(self.microstructure_image[sampleNumber],
                                 (self.img_resolution, self.img_resolution)), cmap='binary')

