import sys
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from file_utils import get_conformers_filenames, get_structures
from acquisition_functions import ExpectedImprovement
from bayesian_utils import (get_interatomic_features, check_convergence,
setup_model_unsupervised_bandwidth, FFTreat)
from initial_samplers import ForceFieldSampler
from conversions import HARTREE_TO_KCAL, HARTREE_TO_JOULES

INIT_SAMPLE_SIZE = 5
SMOOTHING = 0.9
TEMPERATURE = 298.15

def run_optimisation(features, dft_energies, init_sampler):
	model = setup_model_unsupervised_bandwidth(features)
	acq_func = ExpectedImprovement()
	n_samples, seen_indices, unseen_indices = \
		init_sampler.get_sample(dft_energies, INIT_SAMPLE_SIZE)
	score_values = list()
	target_energy = np.nanmin(dft_energies)
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
		print(seen_indices[-1] + 1, HARTREE_TO_KCAL \
			* np.abs(dft_energies[seen_indices[-1]] - target_energy))
		n_samples += 1
	print("Samples = %d" % n_samples)
	print("Proportion = %.3f" % (n_samples / features.shape[0]))
	min_energy = np.min(dft_energies[seen_indices])
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("Min.Energy = %.5f" % min_energy)
	all_energies = dft_energies[~np.isnan(dft_energies)]
	sampled_energies = dft_energies[seen_indices]
	all_energies -= np.min(all_energies)
	sampled_energies -= np.min(sampled_energies)
	all_factors = np.exp(-(HARTREE_TO_JOULES * all_energies) \
		/ (R * TEMPERATURE))
	sampled_factors = np.exp(-(HARTREE_TO_JOULES * sampled_energies) \
		/ (R * TEMPERATURE))
	all_energies *= HARTREE_TO_KCAL
	sampled_energies *= HARTREE_TO_KCAL
	all_boltz = np.sum(all_energies * all_factors) / np.sum(all_factors)
	sampled_boltz = np.sum(sampled_energies * sampled_factors) \
		/ np.sum(sampled_factors)
	boltz_dev = np.abs(all_boltz - sampled_boltz)
	print("Boltz.Dev. = %.5f" % boltz_dev)
	return seen_indices, unseen_indices

def run_experiment(ff_sdf_files, ff_energy_files, dft_energy_files):
	for ff_sdf_file, ff_energy_file, dft_energy_file in zip(ff_sdf_files,
	ff_energy_files, dft_energy_files):
		print("# %s" % ff_sdf_file.replace(".sdf", ""))
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		structures = get_structures(ff_sdf_file, ff_energies)
		features = get_interatomic_features(structures, None, FFTreat.IGNORE)
		init_sampler = ForceFieldSampler(ff_energies)
		seen_indices, unseen_indices = run_optimisation(features, dft_energies,
			init_sampler)
		assert(len(seen_indices) + len(unseen_indices) == features.shape[0])
		dft_energies -= np.min(dft_energies)
		dft_energies *= HARTREE_TO_KCAL
		tsne_features = TSNE(n_components=2,
			random_state=5).fit_transform(features)
		x, y = tsne_features[:,0], tsne_features[:,1]
		x_seen, y_seen = x[seen_indices], y[seen_indices]
		x_unseen, y_unseen = x[unseen_indices], y[unseen_indices]
		plt.scatter(x_seen, y_seen, c=dft_energies[seen_indices], marker='X',
			vmin=np.min(dft_energies), vmax=np.max(dft_energies))
		plt.scatter(x_unseen, y_unseen, c=dft_energies[unseen_indices],
			marker='o', vmin=np.min(dft_energies), vmax=np.max(dft_energies))
		min_index = np.argmin(dft_energies)
		plt.plot([x[min_index]], [y[min_index]], 'x', markersize=10,
			color="black")
		plt.colorbar(label="Relative free energy / kcal/mol")
		plt.tight_layout()
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
