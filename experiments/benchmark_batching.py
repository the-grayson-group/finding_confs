import sys
import numpy as np
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import KrigingBelieverEI
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth)
from initial_samplers import ForceFieldSampler
from conversions import HARTREE_TO_KCAL

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.5

def run_optimisation(features, dft_energies, init_sampler, batch_size):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = KrigingBelieverEI()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
	n_iter = 0
	while len(unseen_indices) > 0:
		acq_func.fit_model(model, features, dft_energies, seen_indices)
		acq_scores = acq_func.get_scores(model, features, unseen_indices)
		score = float(np.mean(acq_scores))
		if len(score_values) > 0:
			new_score = SMOOTHING * score + \
				(1.0 - SMOOTHING) * score_values[-1]
		else:
			new_score = score
		score_values.append(new_score)
		if check_convergence(score_values):
			break
		acq_func.sample_batch(model, features, dft_energies, seen_indices,
			unseen_indices, batch_size)
		n_iter += 1
		if len(unseen_indices) == 0:
			break
	n_samples = features.shape[0] - len(unseen_indices)
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - np.nanmin(dft_energies))
	return n_samples, n_iter, min_energy

def run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files, batch_size):
	total_samples = list()
	total_iters = list()
	proportions = list()
	false_stops = 0
	excess_energies = list()
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		features = get_interatomic_features(structures)
		init_sampler = ForceFieldSampler(ff_energies)
		n_samples, n_iter, excess_energy = run_optimisation(features,
			dft_energies, init_sampler, batch_size)
		total_samples.append(n_samples)
		total_iters.append(n_iter)
		proportions.append(n_samples / features.shape[0])
		false_stops += int(excess_energy != 0.0)
		if excess_energy > 0.0:
			excess_energies.append(excess_energy)
	print("# %d" % batch_size)
	print("TotalSamples = %d" % sum(total_samples))
	print("MeanSamples = %.3f" % np.mean(total_samples))
	print("MaxSamples = %d" % max(total_samples))
	print("TotalIters = %d" % sum(total_iters))
	print("MeanIters = %.3f" % np.mean(total_iters))
	print("MaxIters = %d" % max(total_iters))
	print("MeanProportion = %.4f" % np.mean(proportions))
	print("MaxProportion = %.4f" % max(proportions))
	print("FalseStops = %d" % false_stops)
	if len(excess_energies) > 0:
		print("MeanExcessEnergy = %.4f" % np.mean(excess_energies))
		print("MaxExcessEnergy = %.4f" % max(excess_energies))
	else:
		print("MeanExcessEnergy = 0.0000")
		print("MaxExcessEnergy = 0.0000")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("ERROR Usage: python benchmark_bayesian.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files provided.")
		exit(1)
	for batch_size in (1, 2, 3, 5, 10):
		run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files,
			batch_size)
