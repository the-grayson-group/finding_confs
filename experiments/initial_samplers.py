import numpy as np
from sklearn.cluster import KMeans

class ForceFieldSampler():
	def __init__(self, ff):
		self.ff = ff

	def get_sample(self, y, n_init=5):
		ff_sorted_indices = np.argsort(self.ff)
		i = 0
		n_samples = 0
		seen_indices = list()
		unseen_indices = [i for i in range(len(y))]
		while len(seen_indices) < n_init and len(unseen_indices) > 0:
			sampled_index = ff_sorted_indices[i]
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
	def __init__(self, ff):
		self.ff = ff

	def get_sample(self, y, n_init=5):
		ff_sorted_indices = np.argsort(self.ff)
		n_samples = 0
		seen_indices = list()
		unseen_indices = [i for i in range(len(y))]
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
	def __init__(self, x, seed):
		self.x = x
		self.seed = seed

	def get_sample(self, y, n_init=5):
		kmeans = KMeans(n_clusters=n_init, random_state=self.seed)
		cluster_labels = kmeans.fit_predict(self.x)
		n_samples = 0
		seen_indices = list()
		unseen_indices = [i for i in range(len(y))]
		for i in range(n_init):
			cluster_indices = np.where(cluster_labels == i)[0]
			cluster_points = self.x[cluster_indices]
			cluster_dists = np.sum(np.square(cluster_points - kmeans.cluster_centers_[i]), axis=1)
			sorted_cluster_indices = np.argsort(cluster_dists)
			j = 0
			cluster_index = sorted_cluster_indices[j]
			sampled_index = np.arange(len(y))[cluster_indices][cluster_index]
			unseen_indices.remove(sampled_index)
			n_samples += 1
			j += 1
			while np.isnan(y[sampled_index]) and len(unseen_indices) > 0 and j < len(sorted_cluster_indices):
				cluster_index = sorted_cluster_indices[j]
				sampled_index = np.arange(len(y))[cluster_indices][cluster_index]
				unseen_indices.remove(sampled_index)
				n_samples += 1
				j += 1
			if not np.isnan(y[sampled_index]):
				seen_indices.append(sampled_index)
		return n_samples, seen_indices, unseen_indices
