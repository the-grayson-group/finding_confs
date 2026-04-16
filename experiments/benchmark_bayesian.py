import sys
import numpy as np
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import ExpectedImprovement
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth, FFTreat, FeatureTreat)
from initial_samplers import (ForceFieldSampler, ForceFieldSpreadSampler,
ClusterSampler)
from dihedral_angles import (get_dihedral_angles, filter_dihedral_angles,
process_dihedral_angles)
from conversions import HARTREE_TO_KCAL

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.5

def run_optimisation(features, dft_energies, init_sampler):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = ExpectedImprovement()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
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
		acq_func.process_sample(acq_scores, dft_energies, seen_indices,
			unseen_indices)
	n_samples = features.shape[0] - len(unseen_indices)
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - np.nanmin(dft_energies))
	return n_samples, min_energy

def run_experiment(experiment, ff_sdf_files, ff_energy_files, dft_energy_files):
	feature_treat, initial_sampler, ff_treat = experiment
	total_samples = list()
	proportions = list()
	false_stops = 0
	excess_energies = list()
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		if feature_treat == FeatureTreat.ANGLES:
			features = get_dihedral_angles(structures)
			features = filter_dihedral_angles(features)
			features = process_dihedral_angles(features)
			if ff_treat == FFTreat.INCLUDE:
				features = np.concatenate((features,
					ff_energies.reshape(-1, 1)), axis=1)
			features = features.copy().reshape(features.shape, order="C")
		elif feature_treat == FeatureTreat.DISTS:
			features = get_interatomic_features(structures, ff_energies,
				ff_treat)
		if "Cluster" in repr(initial_sampler):
			init_sampler = initial_sampler(features)
		else:
			init_sampler = initial_sampler(ff_energies)
		n_samples, excess_energy = run_optimisation(features, dft_energies,
			init_sampler)
		total_samples.append(n_samples)
		proportions.append(n_samples / features.shape[0])
		false_stops += int(excess_energy != 0.0)
		if excess_energy > 0.0:
			excess_energies.append(excess_energy)
	print("# %s %s %s" % (feature_treat.name, init_sampler.name,
		ff_treat.name))
	print("TotalSamples = %d" % sum(total_samples))
	print("MeanSamples = %.3f" % np.mean(total_samples))
	print("MaxSamples = %d" % max(total_samples))
	print("MeanProportion = %.4f" % np.mean(proportions))
	print("MaxProportion = %.4f" % max(proportions))
	print("FalseStops = %d" % false_stops)
	if len(excess_energies) > 0:
		print("MeanExcessEnergy = %.4f" % np.mean(excess_energies))
		print("MaxExcessEnergy = %.4f" % max(excess_energies))
	else:
		print("MeanExcessEnergy = 0.0000")
		print("MaxExcessEnergy = 0.0000")

def generate_experiments(params):
	stack = list()
	for i in reversed(range(len(params[0]))):
		stack.append((0, i))
	experiment = list()
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


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("ERROR Usage: python benchmark_bayesian.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files provided.")
		exit(1)
	features = (FeatureTreat.ANGLES, FeatureTreat.DISTS)
	init_samplers = (ForceFieldSampler, ForceFieldSpreadSampler, ClusterSampler)
	ff_treatments = (FFTreat.IGNORE, FFTreat.INCLUDE)
	params = (features, init_samplers, ff_treatments)
	for experiment in generate_experiments(params):
		run_experiment(experiment, ff_sdf_files, ff_energy_files,
			dft_energy_files)
