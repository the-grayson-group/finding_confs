import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

def check_non_monovalent_bond(bond):
	"""
	Returns true if both atoms in a bond have a total valance greater than one.

	Args:
		bond: RDKit Bond object
	"""
	atom1 = bond.GetBeginAtom()
	atom2 = bond.GetEndAtom()
	if atom1.GetTotalValence() == 1 or atom2.GetTotalValence() == 1:
		return False
	return True

def check_non_cx3_bond(bond):
	"""
	Returns true if a bond is not to a CX3 group (X = H, F, Cl, Br or I).

	Args:
		bond: RDKit Bond object
	"""
	atom1 = bond.GetBeginAtom()
	atom2 = bond.GetEndAtom()
	atom1_neighbours = [atom for atom in atom1.GetNeighbors() \
		if atom.GetIdx() != atom2.GetIdx()]
	atom2_neighbours = [atom for atom in atom2.GetNeighbors() \
		if atom.GetIdx() != atom1.GetIdx()]
	x_atom_symbols = ("H", "F", "Cl", "Br", "I")
	for x_atom_symbol in x_atom_symbols:
		atom1_check = all([atom.GetSymbol() == x_atom_symbol \
			for atom in atom1_neighbours])
		atom2_check = all([atom.GetSymbol() == x_atom_symbol \
			for atom in atom2_neighbours])
		if atom1_check or atom2_check:
			return False
	return True

def check_single_bond(bond):
	"""
	Returns true if a single bond is provided.

	Args:
		bond: RDKit Bond object
	"""
	return bond.GetBondTypeAsDouble() == 1.0

def check_non_three_ring_bond(bond, ring_info):
	"""
	Returns true if a bond is not in a three-membered ring.

	Args:
		bond: RDKit Bond object
		ring_info: RDKit RingInfo object
	"""
	bond_idx = bond.GetIdx()
	return not ring_info.IsBondInRingOfSize(bond_idx, 3)

def check_aryl_cx_bond(bond):
	"""
	Returns true if an aromatic C-X bond is provided.

	Args:
		bond: RDKit Bond object
	"""
	symb1 = bond.GetBeginAtom().GetSymbol()
	symb2 = bond.GetEndAtom().GetSymbol()
	has_carbon = symb1 == "C" or symb2 == "C"
	has_hetero = (symb1 != "C" and symb1 != "H") or \
		(symb2 != "C" and symb2 != "H")
	is_aryl = bond.GetBondTypeAsDouble() == 1.5
	if is_aryl and has_carbon and has_hetero:
		return True
	else:
		return False

def check_double_xx_bond(bond):
	"""
	Returns true if a heteroatomic double bond is provided.

	Args:
		bond: RDKit Bond object
	"""
	atom1 = bond.GetBeginAtom()
	atom2 = bond.GetEndAtom()
	symb1 = atom1.GetSymbol()
	symb2 = atom2.GetSymbol()
	have_other_bonds = (len(atom1.GetNeighbors()) > 1) and \
		(len(atom2.GetNeighbors()) > 1)
	bond_order = bond.GetBondTypeAsDouble() == 2.0
	is_xx = (symb1 != "C" and symb1 != "H") and (symb2 != "C" and symb2 != "H")
	if have_other_bonds and bond_order and is_xx:
		return True
	else:
		return False

def check_cn_non_ring_bond(bond):
	"""
	Returns true if a C=N double bond that is not in a ring is provided.

	Args:
		bond: RDKit Bond object
	"""
	atom1 = bond.GetBeginAtom()
	atom2 = bond.GetEndAtom()
	symb1 = atom1.GetSymbol()
	symb2 = atom2.GetSymbol()
	bond_order = bond.GetBondTypeAsDouble() == 2.0
	is_cn = (symb1 == "C" and symb2 == "N") or (symb1 == "N" and symb2 == "C")
	non_ring = not (atom1.IsInRing() and atom2.IsInRing())
	non_all_h_neighbors = check_non_cx3_bond(bond)
	if bond_order and is_cn and non_ring and non_all_h_neighbors:
		return True
	else:
		return False

def check_aryl_cc_double_cy_bond(mol, bond):
	"""
	Returns true if an aromatic C-C(=Y) bond is provided, where Y = O, S or N

	Args:
		bond: RDKit Bond object
	"""
	if bond.GetBondTypeAsDouble() != 1.5:
		return False
	atom1 = bond.GetBeginAtom()
	atom2 = bond.GetEndAtom()
	if not (atom1.GetSymbol() == "C" and atom2.GetSymbol() == "C"):
		return False
	for atom, other_atom in zip((atom1, atom2), (atom2, atom1)):
		for neigh in atom.GetNeighbors():
			# Skip the other atom in the bond
			if neigh.GetIdx() == other_atom.GetIdx():
				continue
			neigh_bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neigh.GetIdx())
			if neigh_bond.GetBondTypeAsDouble() != 2.0:
				continue
			symb = neigh.GetSymbol()
			if symb == "N" or symb == "O" or symb == "S":
				return True
	return False

def get_rotatable_bonds(mol):
	"""
	Gets a list of the rotatable bonds in a molecule, selected according to the
	criteria of the check functions above.

	Args:
		mol: RDKit Mol object

	Returns:
		rotatable_bonds: list of RDKit Bond objects
	"""
	rotatable_bonds = list()
	ring_info = mol.GetRingInfo()
	for bond in mol.GetBonds():
		# Bonds passing the specific checks will be accepted
		aryl_cx = check_aryl_cx_bond(bond)
		double_xx = check_double_xx_bond(bond)
		cn_non_ring = check_cn_non_ring_bond(bond)
		aryl_cc_double_cy = check_aryl_cc_double_cy_bond(mol, bond)
		if aryl_cx or double_xx or cn_non_ring or aryl_cc_double_cy:
			rotatable_bonds.append(bond)
			continue
		# Bonds failing the remaining checks will be rejected
		non_monovalent = check_non_monovalent_bond(bond)
		non_cx3 = check_non_cx3_bond(bond)
		single = check_single_bond(bond)
		non_three_ring = check_non_three_ring_bond(bond, ring_info)
		if non_monovalent and non_cx3 and single and non_three_ring:
			rotatable_bonds.append(bond)
	return rotatable_bonds

def get_atom_score(atom, neigh_atom, ring_info):
	"""
	Score a neighbouring atom to an atom of a rotatable bond. The score is used
	to decide whether that neighbour is used in the bond's dihedral angle
	calculation. A heteroatom gets a base score of 2, carbon gets a base score
	of 1. An atom in the same ring as the rotatable bond atom gets a bonus
	score of 2.

	Args:
		atom: RDKit Atom object in the rotatable bond
		neigh_atom: RDKit Atom object neighbouring the atom in the bond
		ring_info: RDKit RingInfo object

	Returns:
		score: int
	"""
	atomic_num = neigh_atom.GetAtomicNum()
	score = 0
	if atomic_num != 6 and atomic_num != 1:
		score += 2
	elif atomic_num == 6:
		score += 1
	if ring_info.AreAtomsInSameRing(atom.GetIdx(), neigh_atom.GetIdx()):
		score += 2
	return score

def get_dihedral_atoms(mol, bonds):
	"""
	Returns a list of the sets of atom indinces that will be used to calculate
	the dihedral angles of the rotatable bonds in the molecule. The score
	function is used to decide which neighbours of each of the atoms in a bond
	are used in the dihedral.

	Args:
		mol: RDKit Mol object
		bonds: list of RDKit bond objects

	Returns:
		dihedrals: list of 4-tuples of ints giving the atom indices that are
			involved in each rotatable bond
	"""
	dihedrals = list()
	ring_info = mol.GetRingInfo()
	for bond in bonds:
		atom1 = bond.GetBeginAtom()
		atom2 = bond.GetEndAtom()
		atom1_neigh_scores = list()
		atom2_neigh_scores = list()
		max_score = 0
		max_score_atom1 = None
		for atom in atom1.GetNeighbors():
			if atom.GetIdx() == atom2.GetIdx():
				continue
			score = get_atom_score(atom1, atom, ring_info)
			if score > max_score:
				max_score = score
				max_score_atom1 = atom
		max_score = 0
		max_score_atom2 = None
		for atom in atom2.GetNeighbors():
			if atom.GetIdx() == atom1.GetIdx():
				continue
			score = get_atom_score(atom2, atom, ring_info)
			if score > max_score:
				max_score = score
				max_score_atom2 = atom
		assert(max_score_atom1 is not None and max_score_atom2 is not None)
		dihedrals.append((max_score_atom1.GetIdx(), atom1.GetIdx(),
			atom2.GetIdx(), max_score_atom2.GetIdx()))
	return dihedrals

def get_dihedral_angles(structures, extra_atoms=None):
	"""
	Calculate the dihedral angles of all rotatable bonds of all conformers.

	Args:
		structures: list of RDKit mol objects of all the conformers of the
			molecule
		extra_atoms: list of 4-tuples of ints giving the atom indices that are
			involved in any additional rotatable bonds in the molecule that are
			to be considered also

	Returns:
		dihedral_data: numpy array containing dihedral angles of all conformers
	"""
	dihedral_data = list()
	bonds_determined = False
	for mol in structures:
		if not bonds_determined:
			rotatable_bonds = get_rotatable_bonds(mol)
			dihedral_atoms = get_dihedral_atoms(mol, rotatable_bonds)
			if extra_atoms is not None:
				dihedral_atoms.extend(extra_atoms)
			bonds_determined = True
		dihedrals = list()
		conf = mol.GetConformer()
		for atoms in dihedral_atoms:
			dihedrals.append(GetDihedralDeg(conf, *atoms))
		dihedral_data.append(dihedrals)
	dihedral_data = np.array(dihedral_data)
	# Drop any identical columns
	dihedral_data = np.unique(dihedral_data, axis=1)
	return dihedral_data

def filter_dihedral_angles(dihedral_data):
	"""
	Produce a boolean mask array that filters out the dihedral angles that stay
	essentially the same in all the conformers	of a molecule.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.

	Returns:
		filtered_dihedral_data: numpy array containing the dihedral angles of
			the rotatable bonds of each conformer that pass the filter checks.
	"""
	std_check = 2.4
	range_check = 8.7
	abs_std_check = 1.5
	abs_range_check = 6.6
	keep_indices = np.zeros(dihedral_data.shape[1], dtype=bool)
	dihedral_stds = np.std(dihedral_data, axis=0)
	dihedral_ranges = np.ptp(dihedral_data, axis=0)
	angle_check = np.logical_and(dihedral_stds > std_check,
		dihedral_ranges > range_check)
	keep_indices = np.logical_or(keep_indices, angle_check)
	if np.any(dihedral_stds > 359.2):
		abs_dihedral_stds = np.std(np.abs(dihedral_data), axis=0)
		abs_dihedral_ranges = np.ptp(np.abs(dihedral_data), axis=0)
		abs_angle_check = np.logical_and(abs_dihedral_stds > abs_std_check,
			abs_hedral_ranges > abs_range_check)
		abs_angle_check = np.logical_and(abs_angle_check, dihedral_stds > 359.2)
		keep_indices = np.logical_or(keep_indices, abs_angle_check)
	filtered_dihedral_data = dihedral_data[:,keep_indices]
	return filtered_dihedral_data

def process_dihedral_angles(dihedral_data):
	"""
	Post-processes the dihedral angles. This function serves two purposes:
	1. In a dihedral calculation the angles -180 and 180 have a large numerical
	difference, but are actually structurally similar. Therefore, this function
	attempts to correct for this fact. For each conformer in the data set, we
	take all the dihedral angles of one rotatable bond at a time. After shifting
	the angles to the 0-360 range, we check the angles between 0-100
	(-180 - -80) and see if there is a gap of at least 15 degrees between
	two conformers that have the current rotatable bond in this 0-100 range. If
	such a gap exists, the angles on the lower side of it are shifted up by 360
	degrees to match the structually similar high angles.
	2. After the gap-separated, low angles are shifted, the dihedral angles are
	normalised to fit into the 0-1 range.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.

	Returns:
		dihedral_data: numpy array containing the process and normalised
			dihedral angles of each rotatable bond of each conformer.
	"""
	for i in range(dihedral_data.shape[1]):
		angles = dihedral_data[:,i]
		# Shift all angles up to fit the range 0-360
		shifted_angles = angles + 180.0
		# Select low angles below 100 (previouly -80) and search for a gap
		# greater than 15 degrees. If such a gap exists, angles below this gap
		# are moved up by 360 degrees
		low_angles = shifted_angles[shifted_angles<100.0]
		if low_angles.shape[0] > 0:
			low_angles = np.append(low_angles, 100.0)
			low_angles = np.sort(low_angles)
			adjacent_diffs = low_angles[1:] - low_angles[:-1]
			max_gap = np.max(adjacent_diffs)
			if max_gap >= 15.0:
				max_gap_angle = low_angles[np.argmax(adjacent_diffs)]
				# Angles at or below the gap are shifted up to match high angles
				# above the gap.
				shifted_angles[shifted_angles<=max_gap_angle] += 360.0
		dihedral_data[:,i] = shifted_angles
	# Ensure no angles are negative
	dihedral_data -= np.min(dihedral_data, axis=0)
	# Normalise the angle data
	dihedral_data /= np.max(dihedral_data, axis=0)
	return dihedral_data
