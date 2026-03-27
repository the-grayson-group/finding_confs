import os
import sys
import numpy as np
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import ExpectedImprovement
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth)
from initial_samplers import ForceFieldSampler
from conversions import HARTREE_TO_KCAL

import matplotlib.pyplot as plt

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.5

def get_conf_enant_indices(enants_file):
	enant_indices = dict()
	with open(enants_file, "r") as file:
		for line in file:
			line = line.strip().split()
			enant = line[0]
			indices = [int(number) - 1 for number in line[1:]]
			enant_indices[enant] = indices
	return enant_indices

def run_sequential_opt(features, dft_energies, init_sampler, enant_indices):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = ExpectedImprovement()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
	last_seen_len = 0
	# Optimise lowest overall conformer
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
	# To which enantiomer does the lowest energy conformer belong?
	low_index = seen_indices[np.argmin(dft_energies[seen_indices])]
	for enant_key in enant_indices.keys():
		if low_index in enant_indices[enant_key]:
			low_enant = enant_key
		else:
			high_enant = enant_key
	print("%s-Samples = %d" % (low_enant, n_samples))
	print("%s-Proportion = %.3f" % (low_enant, n_samples / features.shape[0]))
	target_energy = np.nanmin(dft_energies)
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("%s-MinEnergy = %.5f" % (low_enant, min_energy))
	# Remove all unseen conformers of the already-optimised, lowest enantiomer
	for index in enant_indices[low_enant]:
		try:
			unseen_indices.remove(index)
		except ValueError:
			continue
	# Get the conformers of the higher enantiomer that have already been seen
	high_seen_indices = list()
	for index in enant_indices[high_enant]:
		if index in seen_indices:
			high_seen_indices.append(index)
	# Repeat the optimisation for the other enantiomer
	score_values = list()
	last_seen_len = 0
	# Optimise lowest conformer of other enantiomer
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
		if check_convergence(score_values, min_thresh=0.1):
			break
		acq_func.process_sample(acq_scores, dft_energies, seen_indices,
			unseen_indices)
		if seen_indices[-1] not in high_seen_indices:
			high_seen_indices.append(seen_indices[-1])
		n_samples += 1
	print("%s-Samples = %d" % (high_enant, n_samples))
	print("%s-Proportion = %.3f" % (high_enant, n_samples / features.shape[0]))
	target_energy = np.nanmin(dft_energies[enant_indices[high_enant]])
	min_energy = np.min(dft_energies[high_seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("%s-MinEnergy = %.5f" % (high_enant, min_energy))

def run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files):
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		mol_name = ff_sdf_file.replace(".sdf", "")
		enants_file = ff_sdf_file.replace(".sdf", "_enants.txt")
		if not os.path.exists(enants_file):
			print("WARNING: No enantiomers file found for %s." % mol_name)
			continue
		print("# %s" % mol_name)
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		features = get_interatomic_features(structures)
		init_sampler = ForceFieldSampler(ff_energies)
		enant_indices = get_conf_enant_indices(enants_file)
		run_sequential_opt(features, dft_energies, init_sampler, enant_indices)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("ERROR Usage: python test_selectivity.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files provided.")
		exit(1)
	run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files)
