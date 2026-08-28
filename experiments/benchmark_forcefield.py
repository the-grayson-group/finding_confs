import sys
import copy
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol
from scipy.constants import R
from file_utils import get_conformers_filenames, get_structures
from forcefield_methods import pipeline_mix_priority_list
from dihedral_angles import (get_dihedral_angles, filter_dihedral_angles,
process_dihedral_angles)
from stop_predictor import calculate_opt_features, evaluate_stop_predictions
from conversions import HARTREE_TO_KCAL, HARTREE_TO_JOULES

TEMPERATURE = 298.15

def check_duplicated_conf(dft_structures, dft_energies, conf_idx, opt_indices):
	"""
	Check if a DFT-optimised conformer is a duplicated of another that is
	already optimised. If the current conformers energy is within 0.1 kcal/mol
	of another, an RMSD calculation is performed, and if the RMSD is lower than
	0.005 Angstrom, then the conformers are considered duplicates and the 
	function returns True.

	Args:
		dft_structures: list of RDKit Mol objects of the DFT-optimised
			conformers
		dft_energies: numpy array containing optimised DFT conformer energies
		conf_idx: int of the index of the current conformer
		opt_indices: list of int of the indices of the conformers that are
			already optimised (excluding missing)
	"""
	rmsd_thresh = 0.005
	energy_thresh = 0.1
	curr_mol = dft_structures[conf_idx]
	if curr_mol is None:
		return False
	curr_mol_copy = copy.copy(curr_mol)
	curr_mol_copy = Chem.RemoveHs(curr_mol_copy, sanitize=False)
	curr_energy = dft_energies[conf_idx]
	for opt_idx in opt_indices:
		opt_mol = dft_structures[opt_idx]
		opt_energy = dft_energies[opt_idx]
		if HARTREE_TO_KCAL * abs(opt_energy - curr_energy) < energy_thresh:
			opt_mol_copy = copy.copy(opt_mol)
			opt_mol_copy = Chem.RemoveHs(opt_mol_copy, sanitize=False)
			atom_map = [(i, i) for i in range(curr_mol_copy.GetNumAtoms())]
			rmsd = AlignMol(curr_mol_copy, opt_mol_copy, atomMap=atom_map)
			if rmsd <= rmsd_thresh:
				return True
	return False

def forcefield_optimise_data(priority_list, dft_energies, dft_structures):
	"""
	Run the optimisation of conformers following the given priority list, at
	each iteration storing the data for a machine learning model that will be
	trained to predict whether the optimisation should stop, having found the
	global minimum energy conformer.

	Args:
		priority_list: list of conformer indices in order of optimisation
			priority
		dft_energies: numpy array containing the DFT energies of each conformer
		dft_structures: list of RDKit Mol objects of the conformers after
			optimisation with DFT

	Returns:
		opt_features: list of lists of optimisation features for each conformer
			in the priority list
		stop_labels: list of labels stating whether or not the global minimum
			conformer has been optimised
		min_energies: list of floats of the minimum energy at each sampling
			iteration
		sampled_energies: list of floats of the conformer energies at each
			iteration
	"""
	opt_features = list()
	stop_labels = list()
	min_energies = list()
	sampled_energies = list()
	chi_new_values = list()
	target_energy = np.nanmin(dft_energies)
	min_energy = float("inf")
	n_confs = len(priority_list)
	for i, conf_idx in enumerate(priority_list):
		curr_energy = dft_energies[conf_idx]
		sampled_energies.append(curr_energy)
		if np.isnan(curr_energy) and len(min_energies) > 0:
			min_energies.append(min_energies[-1])
			continue
		# Calculate chi_new value for the current iteration
		min_energy = min(min_energy, curr_energy)
		min_energies.append(HARTREE_TO_KCAL * (min_energy - target_energy))
		if check_duplicated_conf(dft_structures, dft_energies, conf_idx,
		priority_list[:i]):
			chi_new = 0.0
		else:
			delta_g = HARTREE_TO_JOULES * (curr_energy - min_energy)
			chi_new = np.exp(-delta_g / (R * TEMPERATURE))
		chi_new_values.append(chi_new)
		curr_features = calculate_opt_features(chi_new_values, n_confs)
		opt_features.append(curr_features)
		stop_labels.append(int(min_energy == target_energy))
	assert(len(min_energies) == n_confs)
	return opt_features, stop_labels, min_energies, sampled_energies

def pipeline_mix_opt_data_all(ff_sdf_files, dft_sdf_files, ff_energy_files,
dft_energy_files):
	"""
	For all the molecules provided in a list of files, run the conformer
	energy optimisation with the pipeline-mix method and compile a list of the
	features, stop labels and energies at each iteration.

	Args:
		ff_sdf_files: list of strings of the force field-optimised structure
			sdf files
		dft_sdf_files: list of strings of the DFT-optimised structure sdf files
		ff_energies_files: list of strings of numpy array files containing force
			field energies
		dft_energies_files: list of strings of numpy array files containing DFT
			energies

	Returns:
		opt_features_all: list of lists of lists of floats of the optimistion
			features for each molecule at each sampling iteration
		stop_labels_all: list of lists of ints of the labels indicating whether
			the lowest energy conformers of each molecule has been located at
			each iteration
		min_energies_all: list of lists of floats of the minimum energy found so
			far at each sampling iteration for each molecule
		sampled_energies_all: list of lists of floats of the energies of each
			conformer at each sampling iteration for each molecule
	"""	
	opt_features_all = list()
	stop_labels_all = list()
	min_energies_all = list()
	sampled_energies_all = list()
	for ff_sdf_file, dft_sdf_file, ff_energy_file, dft_energy_file in \
	zip(ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files):
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		ff_structures = get_structures(ff_sdf_file, ff_energies)
		dft_structures = get_structures(dft_sdf_file, dft_energies)
		dihedral_angles = get_dihedral_angles(ff_structures)
		dihedral_angles = filter_dihedral_angles(dihedral_angles)
		dihedral_angles = process_dihedral_angles(dihedral_angles)
		priority_list = pipeline_mix_priority_list(dihedral_angles, ff_energies)
		opt_features, stop_labels, min_energies, sampled_energies = \
			forcefield_optimise_data(priority_list, dft_energies,
			dft_structures)
		opt_features_all.append(opt_features)
		stop_labels_all.append(stop_labels)
		min_energies_all.append(min_energies)
		sampled_energies_all.append(sampled_energies)
	return opt_features_all, stop_labels_all, min_energies_all, \
		sampled_energies_all

def cross_validate_performance(opt_features_all, stop_labels_all,
min_energies_all, sampled_energies_all, mol_names, n_fold=5, confidence=0.9):
	"""
	Cross validates the performance of the pipeline-mix conformer optimisation
	method when stopping determination is performed using a machine learning
	model trained using the training set of each cross-validation fold. Prints
	out performance metrics for all folds combined.

	Args:
		features_all: list of lists of lists of floats of the optimistion
			features for each molecule at each sampling iteration
		labels_all: list of lists of ints of the labels indicating whether the
			lowest energy conformers of each molecule has been located at each
			iteration
		min_energies_all: list of lists of floats of the minimum energy found so
			far at each sampling iteration for each molecule
		sampled_energies_all: list of lists of floats of the energies of each
			conformer at each sampling iteration for each molecule
		mol_names: list of str of the names of each molecule in the benchmark
		n_fold: int giving the number of cross-validation folds to be performed
		confidence: float giving the confidence level above which model
			predictions will be treated as a stop prediction
	"""
	total_samples = list()
	proportions_sampled = list()
	total_false_stops = 0
	excess_energies = list()
	boltz_devs_all = list()
	n_mol = len(opt_features_all)
	n_fold_mol = n_mol // n_fold
	test_start = 0
	test_end = n_fold_mol + 1
	n_remain = n_mol % n_fold - 1
	shuffled_indices = list(range(n_mol))
	random.shuffle(shuffled_indices)
	for k in range(n_fold):
		train_indices = shuffled_indices[:test_start] + \
			shuffled_indices[test_end:]
		test_indices = shuffled_indices[test_start:test_end]
		test_start = test_end
		test_end += n_fold_mol + (1 if n_remain > 0 else 0)
		n_remain -= 1
		train_features = list()
		train_labels = list()
		test_features = list()
		test_labels = list()
		test_min_energies = list()
		test_sampled_energies = list()
		for i, idx in enumerate(train_indices):
			train_features.extend(opt_features_all[idx])
			train_labels.extend(stop_labels_all[idx])
		for i, idx in enumerate(test_indices):
			test_features.extend(opt_features_all[idx])
			test_labels.extend(stop_labels_all[idx])
			test_min_energies.append(min_energies_all[idx])
			test_sampled_energies.append(sampled_energies_all[idx])
		train_features = np.array(train_features)
		train_labels = np.array(train_labels)
		test_features = np.array(test_features)
		test_labels = np.array(test_labels)
		samples, proportions, false_stops, excesses = evaluate_stop_predictions(
			train_features, train_labels, test_features, test_min_energies,
			confidence=confidence)
		total_samples.extend(samples)
		proportions_sampled.extend(proportions)
		total_false_stops += false_stops
		excess_energies.extend([excess for excess in excesses if excess > 0.0])
		boltz_devs = list()
		for n_samples, sampled_energies in zip(samples, test_sampled_energies):
			sampled_energies = np.array(sampled_energies)
			measured_energies = sampled_energies[:n_samples]
			sampled_energies = sampled_energies[~np.isnan(sampled_energies)]
			measured_energies = measured_energies[~np.isnan(measured_energies)]
			sampled_energies -= np.min(sampled_energies)
			measured_energies -= np.min(measured_energies)
			sampled_factors = np.exp(-(HARTREE_TO_JOULES * sampled_energies) \
				/ (R * TEMPERATURE))
			measured_factors = np.exp(-(HARTREE_TO_JOULES * measured_energies) \
				/ (R * TEMPERATURE))
			sampled_energies *= HARTREE_TO_KCAL
			measured_energies *= HARTREE_TO_KCAL
			sampled_boltz = np.sum(sampled_energies * sampled_factors) \
				/ np.sum(sampled_factors)
			measured_boltz = np.sum(measured_energies * measured_factors) \
				/ np.sum(measured_factors)
			boltz_devs.append(np.abs(measured_boltz - sampled_boltz))
		boltz_devs_all.extend(boltz_devs)
		for i, test_index in enumerate(test_indices):
			mol_name = mol_names[test_index]
			print("#", mol_name)
			print("Samples = %d" % samples[i])
			print("Proportion = %.3f" % proportions[i])
			print("Min.Energy = %.5f" % excesses[i])
			print("Boltz.Dev. = %.5f" % boltz_devs[i])
	print("TotalSamples = %d" % sum(total_samples))
	print("MeanSamples = %.3f" % np.mean(total_samples))
	print("MedianSamples = %.3f" % np.median(total_samples))
	print("Std.Samples = %.3f" % np.std(total_samples))
	print("Max.Samples = %d" % max(total_samples))
	print("MeanProportion = %.4f" % np.mean(proportions_sampled))
	print("MedianProportion = %.4f" % np.median(proportions_sampled))
	print("Std.Proportion = %.4f" % np.std(proportions_sampled))
	print("Max.Proportion = %.4f" % max(proportions_sampled))
	print("FalseStops = %d" % total_false_stops)
	if len(excess_energies) > 0:
		print("MeanExcessEnergy = %.4f" % np.mean(excess_energies))
		print("MedianExcessEnergy = %.4f" % np.median(excess_energies))
		print("Std.ExcessEnergy = %.4f" % np.std(excess_energies))
		print("Max.ExcessEnergy = %.4f" % max(excess_energies))
	else:
		print("MeanExcessEnergy = 0.0000")
		print("MedianExcessEnergy = 0.0000")
		print("Std.ExcessEnergy = 0.0000")
		print("Max.ExcessEnergy = 0.0000")
	print("MeanBoltz.Dev. = %.4f" % np.mean(boltz_devs_all))
	print("MedianBoltz.Dev. = %.4f" % np.median(boltz_devs_all))
	print("Std.Boltz.Dev. = %.4f" % np.std(boltz_devs_all))
	print("Max.Boltz.Dev. = %.4f" % max(boltz_devs_all))


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python benchmark_forcefield.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	opt_features_all, stop_labels_all, min_energies_all, \
	sampled_energies_all = pipeline_mix_opt_data_all(ff_sdf_files,
		dft_sdf_files, ff_energy_files, dft_energy_files)
	mol_names = [filename.replace(".sdf", "") for filename in ff_sdf_files]
	for confidence in (0.6, 0.7, 0.8, 0.9):
		random.seed(5)
		print("# %d%%" % int(100 * confidence))
		cross_validate_performance(opt_features_all, stop_labels_all,
			min_energies_all, sampled_energies_all, mol_names, n_fold=5,
			confidence=confidence)
