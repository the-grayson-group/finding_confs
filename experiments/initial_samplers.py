import numpy as np
from sklearn.cluster import AgglomerativeClustering

class ForceFieldSampler():
	"""
	Initial sampling method which uses the conformers with the lowest force
	field energies as the initial sample.
	"""
	def __init__(self, ff):
		self.ff = ff
		self.name = "FFLow"

	def get_sample(self, y, n_init):
		ff_sorted_indices = np.argsort(self.ff)
		i = 0
		n_samples = 0
		seen_indices = list()
		unseen_indices = list(range(len(y)))
		while len(seen_indices) < n_init and len(unseen_indices) > 0:
			sampled_index = int(ff_sorted_indices[i])
			unseen_indices.remove(sampled_index)
			n_samples += 1
			i += 1
			while np.isnan(y[sampled_index]) and len(unseen_indices) > 0:
				sampled_index = int(ff_sorted_indices[i])
				unseen_indices.remove(sampled_index)
				n_samples += 1
				i += 1
			if not np.isnan(y[sampled_index]):
				seen_indices.append(sampled_index)
		return n_samples, seen_indices, unseen_indices

class ForceFieldSpreadSampler():
	"""
	Initial sampling method which uses conformers with an even spread of force
	field energies from low to high.
	"""
	def __init__(self, ff):
		self.ff = ff
		self.name = "FFSpread"

	def get_sample(self, y, n_init):
		ff_sorted_indices = np.argsort(self.ff)
		n_samples = 0
		seen_indices = list()
		unseen_indices = list(range(len(y)))
		selected_indices = [i * (len(y) - 1) // n_init for i in range(n_init)]
		for i in range(n_init):
			sampled_index = int(ff_sorted_indices[selected_indices[i]])
			if len(unseen_indices) <= 0:
				return n_samples, seen_indices, unseen_indices
			unseen_indices.remove(sampled_index)
			n_samples += 1
			selected_indices[i] += 1
			while np.isnan(y[sampled_index]) and len(unseen_indices) > 0:
				if selected_indices[i] >= len(y):
					return n_samples, seen_indices, unseen_indices
				sampled_index = int(ff_sorted_indices[selected_indices[i]])
				unseen_indices.remove(sampled_index)
				n_samples += 1
				selected_indices[i] += 1
			if not np.isnan(y[sampled_index]):
				seen_indices.append(sampled_index)
		return n_samples, seen_indices, unseen_indices

class ClusterSampler():
	"""
	Initial sampling method that splits the conformers into a number of clusters
	and selects the most central point from each cluster as a member of the
	initial sample.
	"""
	def __init__(self, x):
		self.x = x
		self.name = "Cluster"

	def get_sample(self, y, n_init):
		clustering = AgglomerativeClustering(n_clusters=n_init)
		cluster_labels = clustering.fit_predict(self.x)
		n_samples = 0
		seen_indices = list()
		unseen_indices = list(range(len(y)))
		all_indices = np.arange(len(y))
		for i in range(n_init):
			cluster_indices = np.where(cluster_labels == i)[0]
			cluster_points = self.x[cluster_indices]
			cluster_centre = np.mean(cluster_points, axis=0)
			cluster_dists = np.sum(np.square(cluster_points - cluster_centre),
				axis=1)
			sorted_cluster_indices = np.argsort(cluster_dists)
			j = 0
			cluster_index = sorted_cluster_indices[j]
			sampled_index = int(all_indices[cluster_indices][cluster_index])
			unseen_indices.remove(sampled_index)
			n_samples += 1
			j += 1
			while np.isnan(y[sampled_index]) and len(unseen_indices) > 0 and \
			j < len(sorted_cluster_indices):
				cluster_index = sorted_cluster_indices[j]
				sampled_index = int(all_indices[cluster_indices][cluster_index])
				unseen_indices.remove(sampled_index)
				n_samples += 1
				j += 1
			if not np.isnan(y[sampled_index]):
				seen_indices.append(sampled_index)
		return n_samples, seen_indices, unseen_indices
