import os
import sys
import copy
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol
from scipy.constants import R
from file_utils import get_conformers_filenames, get_structures
from forcefield_methods import pipeline_mix_priority_list
from dihedral_angles import (get_dihedral_angles, filter_dihedral_angles,
process_dihedral_angles)
from stop_predictor import calculate_opt_features
from conversions import HARTREE_TO_KCAL, HARTREE_TO_JOULES

def get_extra_atoms(extra_atoms_file):
	"""
	Read in a file containing (1-indexed) atom indices that give any additional
	rotatable bonds in a molecule that should be included in its dihedral angle
	features, but may not already be considered (e.g. breaking/forming bonds in
	transition states).

	Args:
		extra_atoms_file: string providing path to file containing extra atoms
			information

	Returns:
		extra_atoms: list of 4-tuples containing indices of the atoms involved
			in the additional rotatable bonds
	"""
	extra_atoms = list()
	with open(extra_atoms_file) as file:
		for line in file:
			atoms = tuple(int(num) - 1 for num in line.split())
			assert(len(atoms) == 4)
			extra_atoms.append(atoms)
	return extra_atoms

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

def forcefield_optimise_data(priority_list, dft_energies, dft_structures,
stop_predictor, confidence):
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
		stop_predictor: sklearn LogisticRegression model trained to predict
			when the lowest energy conformer has been located using the
			pipeline-mix method
		confidence: float giving the stop predictor's predicted probability
			must be above for the optimisation to terminate
	"""
	stop_idx = np.where(stop_predictor.classes_ == 1.0)[0][0]
	chi_new_values = list()
	min_energy = float("inf")
	temperature = 298.15
	n_confs = len(priority_list)
	n_samples = 0
	stop_predictions = list()
	n_points = 3
	for i, conf_idx in enumerate(priority_list):
		n_samples += 1
		curr_energy = dft_energies[conf_idx]
		if np.isnan(curr_energy):
			continue
		# Calculate chi_new value for the current iteration
		if check_duplicated_conf(dft_structures, dft_energies, conf_idx,
		priority_list[:i]):
			chi_new = 0.0
		else:
			delta_g = HARTREE_TO_JOULES * (curr_energy - min_energy)
			chi_new = np.exp(-delta_g / (R * temperature))
		chi_new_values.append(chi_new)
		min_energy = min(min_energy, curr_energy)
		curr_features = calculate_opt_features(chi_new_values, n_confs)
		stop_pred = stop_predictor.predict_proba([curr_features])[:,stop_idx]
		stop_predictions.append(stop_pred)
		if all(pred >= confidence for pred in stop_predictions[-n_points:]) \
		and len(stop_predictions) > n_points:
			break
	print("Samples = %d" % n_samples)
	print("Proportion = %.3f" % (n_samples / len(priority_list)))
	target_energy = np.nanmin(dft_energies)
	min_energy = HARTREE_TO_KCAL * (min_energy - target_energy)
	print("MinEnergy = %.5f" % min_energy)

def pipeline_mix_optimise(ff_sdf_files, dft_sdf_files, ff_energy_files,
dft_energy_files, stop_predictor, confidence=0.9):
	"""
	For all the molecules provided in a list of files, run the conformer
	energy optimisation with the pipeline-mix method and use a pre-trained stop
	predictor model to terminate the optimisation.

	Args:
		ff_sdf_files: list of strings of the force field-optimised structure
			sdf files
		dft_sdf_files: list of strings of the DFT-optimised structure sdf files
		ff_energies_files: list of strings of numpy array files containing force
			field energies
		dft_energies_files: list of strings of numpy array files containing DFT
			energies
		stop_predictor: sklearn LogisticRegression model trained to predict
			when the lowest energy conformer has been located using the
			pipeline-mix method
		confidence: float giving the stop predictor's predicted probability
			must be above for the optimisation to terminate
	"""	
	for ff_sdf_file, dft_sdf_file, ff_energy_file, dft_energy_file in \
	zip(ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files):
		print("# %s" % ff_sdf_file.replace(".sdf", ""))
		ff_energies = np.load(ff_energy_file)
		dft_energies = np.load(dft_energy_file)
		ff_structures = get_structures(ff_sdf_file, ff_energies)
		dft_structures = get_structures(dft_sdf_file, dft_energies)
		tsbonds_file = ff_sdf_file.replace(".sdf", "_tsbonds.txt")
		if os.path.exists(tsbonds_file):
			extra_atoms = get_extra_atoms(tsbonds_file)
		else:
			extra_atoms = None
		dihedral_angles = get_dihedral_angles(ff_structures, extra_atoms)
		dihedral_angles = filter_dihedral_angles(dihedral_angles)
		dihedral_angles = process_dihedral_angles(dihedral_angles)
		priority_list = pipeline_mix_priority_list(dihedral_angles, ff_energies)
		forcefield_optimise_data(priority_list, dft_energies, dft_structures,
			stop_predictor, confidence)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python benchmark_forcefield.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files provided.")
		exit(1)
	stop_predictor_filename = "molecules_stop_pred.pkl"
	if not os.path.exists(stop_predictor_filename):
		print("ERROR: Stop predictor %s not found." % stop_predictor_filename)
		exit(1)
	stop_predictor_file = open(stop_predictor_filename, "rb")
	stop_predictor = pickle.load(stop_predictor_file)
	pipeline_mix_optimise(ff_sdf_files, dft_sdf_files, ff_energy_files,
		dft_energy_files, stop_predictor, confidence=0.8)
