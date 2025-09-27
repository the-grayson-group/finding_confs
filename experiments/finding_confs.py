import os
import sys
import ctypes
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from acquisition_functions import Exploitation, ProbabilityImprovement, ExpectedImprovement, LowerConfidenceBound
from initial_samplers import ForceFieldSampler, ForceFieldSpreadSampler, ClusterSampler

def generate_experiments():
	acq_func = (0, 1, 2, 3)
	init_scheme = (0, 1, 2)
	ff_treat = (0, 1, 2)
	n_dims = (3, 4, 6, 10, 20, 30)
	params = (acq_func, init_scheme, ff_treat, n_dims)
	stack = []
	for i in reversed(range(len(params[0]))):
		stack.append((0, i))
	experiment = []
	while len(stack) > 0:
		level, index = stack.pop()
		while len(experiment) > level:
			experiment.pop()
		experiment.append(params[level][index])
		if len(experiment) == len(params):
			yield experiment
			experiment.pop()
		elif level < len(params) - 1:
			for i in reversed(range(len(params[level+1]))):
				stack.append((level + 1, i))

def convert_expts_to_objs(acq, init):
	acq_funcs = (Exploitation, LowerConfidenceBound, ProbabilityImprovement, ExpectedImprovement)
	acq_func = acq_funcs[acq]
	samplers = (ForceFieldSampler, ForceFieldSpreadSampler, ClusterSampler)
	sampler = samplers[init]
	return acq_func, sampler

def minimise_conf_energy(x, y, seen_indices, unseen_indices, model, acq_func, n_samples, results_file):
	target_y = np.min(y[np.where(~np.isnan(y))])
	while len(unseen_indices) > 0:
		min_y = 627.509 * (np.min(y[seen_indices]) - target_y)
		last_y = 627.509 * (y[seen_indices[-1]] - target_y)
		results_file.write(f"{n_samples} {min_y:.5f} {last_y:.5f}\n")
		sampled_index = acq_func.select_next(x, y, seen_indices, unseen_indices, model, True)
		n_samples = acq_func.process_sample(sampled_index, x, y, seen_indices, unseen_indices, model, n_samples)
	min_y = 627.509 * (np.min(y[seen_indices]) - target_y)
	last_y = 627.509 * (y[seen_indices[-1]] - target_y)
	results_file.write(f"{n_samples} {min_y:.5f} {last_y:.5f}\n")

def run_seeded_experiment(expt, features, dft_energies, ff_energies, seed, results_file):
	acq, init, ff_treat, dims = expt
	results_file.write(f"# {acq} {init} {ff_treat} {dims}\n")
	acq_func, sampler = convert_expts_to_objs(acq, init)
	acq_func = acq_func()
	dims = min(dims, features.shape[0], features.shape[1])
	pca = PCA(n_components=dims, random_state=seed)
	if ff_treat == 1:
		features = np.concatenate((features, ff_energies.reshape(-1, 1)), axis=1)
	features = StandardScaler().fit_transform(features)
	features = pca.fit_transform(features)
	if ff_treat == 2:
		features = np.concatenate((features, ff_energies.reshape(-1, 1)), axis=1)
	features = StandardScaler().fit_transform(features)
	lib = ctypes.CDLL("./libgeometry.so")
	lib.rbf_maximise_variance.restype = ctypes.c_double
	features = features.copy().reshape(features.shape, order="C")
	len_scale = lib.rbf_maximise_variance(np.ctypeslib.as_ctypes(features), ctypes.c_size_t(features.shape[0]), ctypes.c_size_t(features.shape[1]), ctypes.c_int(3), ctypes.c_int(-6))
	kernel = WhiteKernel(noise_level=0.01) + RBF(length_scale=len_scale, length_scale_bounds="fixed")
	model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, random_state=seed)
	if init == 2:
		sampler_inst = sampler(features, seed)
	else:
		sampler_inst = sampler(ff_energies)
	n_samples, seen_indices, unseen_indices = sampler_inst.get_sample(dft_energies)
	minimise_conf_energy(features, dft_energies, seen_indices, unseen_indices, model, acq_func, n_samples, results_file)

def run_experiments(feature_file, dft_file, ff_file, n_repeats, results_file):
	molecule_name = feature_file.replace("_data.npy", "")
	features = np.load(feature_file)
	dft_energies = np.load(dft_file)
	ff_energies = np.load(ff_file)
	features = VarianceThreshold(0.0001).fit_transform(features)
	experiments = generate_experiments()
	for expt in experiments:
		for seed in range(1, n_repeats + 1):
			np.random.seed(seed)
			results_file.write(f"# {molecule_name} {seed}\n")
			run_seeded_experiment(expt, features, dft_energies, ff_energies, seed, results_file)

def get_data_cmdline():
	feature_data_files = list()
	dft_data_files = list()
	ff_data_files = list()
	for filename in sys.argv[1:]:
		if os.path.isdir(filename):
			for data_filename in os.listdir(filename):
				if data_filename.endswith("_data.npy"):
					data_filename = os.path.join(filename, data_filename)
					dft_filename = data_filename.replace("_data.npy", "_dft.npy")
					ff_filename = data_filename.replace("_data.npy", "_ff.npy")
					if os.path.exists(dft_filename) and os.path.exists(ff_filename):
						feature_data_files.append(data_filename)
						dft_data_files.append(dft_filename)
						ff_data_files.append(ff_filename)
					else:
						continue
		elif filename.endswith("_data.npy"):
			dft_filename = filename.replace("_data.npy", "_dft.npy")
			ff_filename = filename.replace("_data.npy", "_ff.npy")
			if os.path.exists(dft_filename) and os.path.exists(ff_filename):
				feature_data_files.append(filename)
				dft_data_files.append(dft_filename)
				ff_data_files.append(ff_filename)
	return feature_data_files, dft_data_files, ff_data_files

if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("ERROR: Need to provide data files.")
		exit(1)
	feature_files, dft_files, ff_files = get_data_cmdline()
	n_repeats = 25
	results_file = open(os.path.dirname(sys.argv[1]) + "_find_conf_results.txt", "w")
	for i in range(len(feature_files)):
		run_experiments(feature_files[i], dft_files[i], ff_files[i], n_repeats, results_file)
	results_file.close()
