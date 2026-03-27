import ctypes
from enum import Enum
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class FFTreat(Enum):
	IGNORE = 0
	INCLUDE = 1

class FeatureTreat(Enum):
	ANGLES = 0
	DISTS = 1

def get_interatomic_features(structures, ff_energies=None,
ff_treat=FFTreat.IGNORE):
	"""
	Calculate the reciprocal heavy-atom distance matrix features for each
	conformer of a molecule.

	Args:
		structures: list of RDKit mol objects giving the conformer geometries
		ff_energies: numpy array containing the force field energies of the
			conformers of the molecule
		ff_treat: FFTreat enum specifying how the force field energies should
			be treated as a feature

	Returns:
		features: 2D numpy array containing the PCA-compressed reciprocal heavy-
		atom distance matrix features for each conformer of a molecule
	"""
	features = list()
	lib = ctypes.CDLL("./libgeometry.so")
	mol = structures[0]
	n_atoms = mol.GetNumAtoms()
	n_dists = (n_atoms * (n_atoms - 1)) // 2
	n_atoms_inp = ctypes.c_size_t(n_atoms)
	n_dists_inp = ctypes.c_size_t(n_dists)
	for mol in structures:
		conf = mol.GetConformer()
		coords = conf.GetPositions()
		values = np.empty(n_dists)
		coords_inp = np.ctypeslib.as_ctypes(coords)
		values_inp = np.ctypeslib.as_ctypes(values)
		lib.calculate_inv_dists(coords_inp, n_atoms_inp, values_inp,
			n_dists_inp)
		features.append(values)
	features = np.array(features)
	features = StandardScaler().fit_transform(features)
	pca = PCA(n_components=0.99, svd_solver="full")
	features = pca.fit_transform(features)
	if ff_treat == FFTreat.INCLUDE and ff_energies is not None:
		ff_energies = np.reshape(ff_energies, (-1, 1))
		ff_energies = StandardScaler().fit_transform(ff_energies)
		features = np.concatenate((features, ff_energies), axis=1)
	features = features.copy().reshape(features.shape, order="C")
	return features

def setup_model_unsupervised_bandwidth(features):
	"""
	Calculate the bandwidth/length scale hyperparameter for the RBF kernel
	function using the variance-maximising value of that hyperparameter and
	return an initialised model using this RBF kernel plus a fixed noise
	component.

	Args:
		features: 2D numpy array containing the features representing the
			geometry of each of conformer of the molecule

	Returns:
		model: sklearn GaussianProcessRegressor model with RBF + WhiteKernel
			kernel function, where the RBF length scale is determined by the
			value that maximises the variance of the data in the RBF kernel
			space
	"""
	lib = ctypes.CDLL("./libgeometry.so")
	lib.rbf_maximise_variance.restype = ctypes.c_double
	features_inp = np.ctypeslib.as_ctypes(features)
	feature_rows_inp = ctypes.c_size_t(features.shape[0])
	feature_cols_inp = ctypes.c_size_t(features.shape[1])
	u_inp = ctypes.c_int(3)
	l_inp = ctypes.c_int(-6)
	kernel_len_scale = lib.rbf_maximise_variance(features_inp, feature_rows_inp,
		feature_cols_inp, u_inp, l_inp)
	if kernel_len_scale <= 0.0:
		print("ERROR Failed to calculate kernel bandwidth")
		exit(1)
	kernel = RBF(length_scale=kernel_len_scale, length_scale_bounds="fixed")
	model = GaussianProcessRegressor(kernel=kernel, optimizer=None,
		normalize_y=True)
	return model

def check_convergence(acq_values, min_thresh=0.01, grad_thresh=0.0001):
	"""
	Check for convergence in the acquisition function values of the Bayesian
	optimisation. Returns true if the acqusition function value has consistently
	reduced to become less than a given percentage of its maximum value and a
	three-point backwards finite differences approximation of the gradient of
	the acquisition function values has converged.

	Args:
		acq_values: list of floats of the acquisition function values up to the
			current iteration
		min_thresh: float giving the percentage of the overall maximum
			acquisition function value that the acquisition function should be
			below for convergence
		grad_thresh: float giving the tolerance that the gradient of the
			acquisition function values must be below for convergence

	Returns: bool
	"""
	n_points = 3
	if len(acq_values) < n_points:
		return False
	max_val = max(acq_values)
	prev_vals = acq_values[-n_points:]
	check1 = all(value < min_thresh * max_val for value in prev_vals)
	grad_est = (prev_vals[0] - 4.0 * prev_vals[1] + 3.0 * prev_vals[2]) / 2.0
	check2 = abs(grad_est) < grad_thresh
	if check1 and check2:
		return True
	return False
