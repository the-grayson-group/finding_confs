import os
import sys
import numpy as np
from rdkit import Chem

def get_conformers_filenames():
	"""
	Finds sdf and numpy array files containing conformer energies from the
	command line.
	Reads each filename from the command line in turn. If a folder is found,
	will find all *.sdf, *_dft.sdf, *_ff.npy and *_dft.npy files in that folder.
	If a .sdf file is found, will find the corresponding _dft.sdf, _ff.npy and
	_dft.npy files also.

	Returns:
		ff_sdf_files: list of sdf files of conformers from the force field
		dft_sdf_files: list of sdf files of conformers optimised with DFT
		ff_energies_files: list of numpy array files of force field energies
		dft_energies_files: list of numpy array files of dft energies
	"""
	ff_sdf_files = list()
	dft_sdf_files = list()
	ff_energies_files = list()
	dft_energies_files = list()
	for filename in sys.argv[1:]:
		if not os.path.exists(filename):
			continue
		if os.path.isdir(filename):
			for conf_filename in os.listdir(filename):
				if conf_filename.endswith(".sdf"):
					sdf_filename = os.path.join(filename, conf_filename)
					dft_sdf_filename = sdf_filename.replace(".sdf", "_dft.sdf")
					ff_filename = sdf_filename.replace(".sdf", "_ff.npy")
					dft_filename = sdf_filename.replace(".sdf", "_dft.npy")
					if os.path.exists(dft_filename) and \
					os.path.exists(ff_filename) and \
					os.path.exists(dft_sdf_filename):
						ff_sdf_files.append(sdf_filename)
						dft_sdf_files.append(dft_sdf_filename)
						ff_energies_files.append(ff_filename)
						dft_energies_files.append(dft_filename)
		elif filename.endswith(".sdf") and not filename.endswith("_dft.sdf"):
			dft_sdf_filename = filename.replace(".sdf", "_dft.sdf")
			ff_filename = filename.replace(".sdf", "_ff.npy")
			dft_filename = filename.replace(".sdf", "_dft.npy")
			if os.path.exists(dft_filename) and os.path.exists(ff_filename) and\
			os.path.exists(dft_sdf_filename):
				ff_sdf_files.append(filename)
				dft_sdf_files.append(dft_sdf_filename)
				ff_energies_files.append(ff_filename)
				dft_energies_files.append(dft_filename)
	return ff_sdf_files, dft_sdf_files, ff_energies_files, dft_energies_files

def get_structures(sdf_filename, energies):
	"""
	Creates a list of RDKit molecules, with None where the energies in the
	provided array are missing.

	Args:
		sdf_filename: string of the filename for the conformers
		energies: numpy array containing the energies of each conformer
			(NaN where conformer energy is missing)

	Returns:
		mols: list of RDKit Mol objects of the conformers
	"""
	mols = list()
	suppl = Chem.SDMolSupplier(sdf_filename, sanitize=False, removeHs=False)
	for i in range(energies.shape[0]):
		if not np.isnan(energies[i]):
			mol = next(suppl)
			mols.append(mol)
		else:
			mols.append(None)
	return mols
