import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import ExpectedImprovement
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth, FFTreat)
from initial_samplers import ForceFieldSampler
from conversions import HARTREE_TO_KCAL

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.9

def get_model_accuracy(model, features, dft_energies, seen_indices,
unseen_indices, acq_func):
	train_features = features[seen_indices]
	train_energies = dft_energies[seen_indices]
	test_features = features[unseen_indices]
	test_energies = dft_energies[unseen_indices]
	train_preds = model.predict(train_features)
	train_preds = train_preds * acq_func.y_std + acq_func.y_mean
	test_preds = model.predict(test_features)
	test_preds = test_preds * acq_func.y_std + acq_func.y_mean
	train_error = np.mean(HARTREE_TO_KCAL \
		* np.abs(train_preds - train_energies))
	valid_mask = np.logical_not(np.isnan(test_energies))
	test_energies = test_energies[valid_mask]
	test_preds = test_preds[valid_mask]
	test_error = np.mean(HARTREE_TO_KCAL * np.abs(test_preds - test_energies))
	return train_error, test_error

def run_optimisation(features, dft_energies, init_sampler):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = ExpectedImprovement()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
	data_sizes = list()
	train_errors = list()
	test_errors = list()
	while len(unseen_indices) > 0:
		acq_func.fit_model(model, features, dft_energies, seen_indices)
		train_error, test_error = get_model_accuracy(model, features,
			dft_energies, seen_indices, unseen_indices, acq_func)
		data_sizes.append(features.shape[0] - len(unseen_indices))
		train_errors.append(train_error)
		test_errors.append(test_error)
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
		acq_func.process_sample(acq_scores, dft_energies, seen_indices,
			unseen_indices)
		n_samples += 1
	print("Samples = %d" % n_samples)
	print("Proportion = %.3f" % (n_samples / features.shape[0]))
	target_energy = np.nanmin(dft_energies)
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("MinEnergy = %.5f" % min_energy)
	return data_sizes, train_errors, test_errors

def run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files):
	i = 0
	name_map = {"BPA_TS": "BPA", "NTOB_DA": "NTOB", "adsbimp": "BIMP"}
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		print("# %s" % ff_sdf_file.replace(".sdf", ""))
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		features = get_interatomic_features(structures, None, FFTreat.IGNORE)
		init_sampler = ForceFieldSampler(ff_energies)
		data_sizes, train_errors, test_errors = run_optimisation(features,
			dft_energies, init_sampler)
		label = os.path.basename(ff_sdf_file).replace(".sdf", "")
		if name_map.get(label):
			label = name_map[label]
		plt.plot(data_sizes, train_errors, "o-", color="C" + str(i))
		plt.plot(data_sizes, test_errors, "x-", color="C" + str(i), label=label)
		i += 1
	plt.legend()
	plt.title("GPR Accuracy As Data Collected", fontsize=14)
	plt.xlabel("Number of Conformers Optimized", fontsize=14)
	plt.ylabel("GPR MAE / kcal/mol", fontsize=14)
	plt.show()


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
