import numpy as np
from sklearn.cluster import AgglomerativeClustering

def label_k_clusters(dihedral_data, k):
	"""
	Use agglomerative clustering with a given number of clusters, k, to cluster
	and label the conformers in a dataset based on the dihedral angles of their
	rotatable bonds.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.
		k: int of the number of clusters

	Returns:
		cluster_labels: numpy array containing which cluster each conformer
			belongs to
	"""
	cluster_model = AgglomerativeClustering(n_clusters=k, metric="euclidean",
		linkage="ward")
	cluster_labels = cluster_model.fit_predict(dihedral_data)
	return cluster_labels

def pipeline_x_priority_list(dihedral_data, ff_energies, x=0.8):
	"""
	Creates a priority list of conformers to be optimised using the pipline-x
	method, which selects a number of clusters (x multiplied by the number of
	conformers) and the lowest energy conformer from each cluster in turn is
	prioritised for optimisation.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.
		ff_energies: number array containing low-level (force field) energies
			of each of the conformers
		x: float value between 0 and 1 determining the number clusters as a
			proportion of the number of conformers

	Returns:
		priority_list: list of conformer indices in order of optimisation
			priority
	"""
	priority_list = list()
	n_clusters = int(x * dihedral_data.shape[0])
	cluster_labels = label_k_clusters(dihedral_data, n_clusters)
	# Make a list of lists of indices of the conformers that belong to each
	# cluster
	cluster_conf_indices = [list(int(idx) \
		for idx in np.where(cluster_labels == i)[0]) for i in range(n_clusters)]
	# Sort the conformer indices in each cluster by the force field energy
	cluster_conf_indices = [list(sorted(conf_indices,
		key=lambda idx: ff_energies[idx])) \
		for conf_indices in cluster_conf_indices]
	max_n_confs = max(len(conf_indices) \
		for conf_indices in cluster_conf_indices)
	for i in range(max_n_confs):
		# Extract the lowest energy conformers of each cluster
		lowest_conf_indices = [conf_indices[i] \
			for conf_indices in cluster_conf_indices if i < len(conf_indices)]
		# Sort the selected conformers from each cluster by force field energy
		lowest_conf_indices = list(sorted(lowest_conf_indices,
			key=lambda idx: ff_energies[idx]))
		priority_list.extend(lowest_conf_indices)
	return priority_list

def pipeline_ascent_priority_list(dihedral_data, ff_energies):
	"""
	Creates a priority list of conformers to be optimised using the pipeline-
	ascent method, which iterates a number of clusters from 1 to the total
	number of conformers and adds the lowest energy conformer from each cluster
	to the priority list at each iteration.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.
		ff_energies: number array containing low-level (force field) energies
			of each of the conformers

	Returns:
		priority_list: list of conformer indices in order of optimisation
			priority
	"""
	priority_list = [int(np.argmin(ff_energies))]
	for k in range(2, dihedral_data.shape[0] + 1):
		cluster_labels = label_k_clusters(dihedral_data, k)
		# Get a list of lists of which conformer indices belong to each cluster
		cluster_conf_indices = [list(int(idx) \
			for idx in np.where(cluster_labels == i)[0]) for i in range(k)]
		# Get the lowest energy conformers of each cluster
		min_conf_indices = [min(conf_indices, key=lambda idx: ff_energies[idx])\
			for conf_indices in cluster_conf_indices]
		# Filter out conformers that have already been selected
		min_conf_indices = list(filter(lambda idx: idx not in priority_list,
			min_conf_indices))
		assert(len(min_conf_indices) == 1)
		priority_list.extend(min_conf_indices)
	return priority_list

def pipeline_mix_priority_list(dihedral_data, ff_energies, q=0.2, x=0.8):
	"""
	Creates a priority list of conformers to be optimised using the pipeline-mix
	method, which takes the a proportion of the first conformers from the
	pipeline-x method (determined by the q parameter) and the remaining
	conformers are ordered using the pipeline-ascent method.

	Args:
		dihedral_data: numpy array containing the dihedral angles of each
			rotatable bond in the molecule conformers. Rows correspond to
			conformers, columns to the rotatable bonds.
		ff_energies: number array containing low-level (force field) energies
			of each of the conformers
		q: float value between 0 and 1 determining the proportion of conformers
			that will be selected using the pipeline-x method
		x: float value between 0 and 1 determining the number clusters as a
			proportion of the number of conformers for the pipeline-x method

	Returns:
		priority_list: list of conformer indices in order of optimisation
			priority
	"""
	x_priority_list = pipeline_x_priority_list(dihedral_data, ff_energies, x)
	ascent_priority_list = pipeline_ascent_priority_list(dihedral_data,
		ff_energies)
	n_confs_x = int(q * dihedral_data.shape[0])
	priority_list = x_priority_list[:n_confs_x]
	priority_list.extend([idx for idx in ascent_priority_list \
		if idx not in priority_list])
	return priority_list
