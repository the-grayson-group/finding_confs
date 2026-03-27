import sys
import numpy as np
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import ExpectedImprovement
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth, FFTreat)
from initial_samplers import ForceFieldSampler
from conversions import HARTREE_TO_KCAL

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.5

def run_optimisation(features, dft_energies, init_sampler):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = ExpectedImprovement()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
	last_seen_len = 0
	while len(unseen_indices) > 0:
		acq_func.fit_model(model, features, dft_energies, seen_indices)
		acq_scores = acq_func.get_scores(model, features, unseen_indices)
		if len(seen_indices) > last_seen_len:
			score = float(np.mean(acq_scores))
			if len(score_values) > 0:
				new_score = SMOOTHING * score + \
					(1.0 - SMOOTHING) * score_values[-1]
			else:
				new_score = score
			score_values.append(new_score)
		last_seen_len = len(seen_indices)
		if check_convergence(score_values):
			break
		acq_func.process_sample(acq_scores, dft_energies, seen_indices,
			unseen_indices)
		n_samples += 1
	print("Samples = %d" % n_samples)
	print("Proportion = %.3f" % (n_samples / features.shape[0]))
	target_energy = np.nanmin(dft_energies)
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("MinEnergy = %.5f" % min_energy)

def run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files):
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		print("# %s" % ff_sdf_file.replace(".sdf", ""))
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		features = get_interatomic_features(structures, None, FFTreat.IGNORE)
		init_sampler = ForceFieldSampler(ff_energies)
		run_optimisation(features, dft_energies, init_sampler)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("ERROR Usage: python benchmark_bayesian.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files provided.")
		exit(1)
	run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files)
