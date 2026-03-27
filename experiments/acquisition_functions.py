import numpy as np
from scipy.stats import norm

class BaseAcquisitionFunction():
	def fit_model(self, model, x, y, seen_indices):
		x_seen = x[seen_indices]
		y_seen = y[seen_indices]
		y_mean, y_std = y_seen.mean(), y_seen.std()
		y_seen = (y_seen - y_mean) / y_std
		model.fit(x_seen, y_seen)
		self.y_mean = y_mean
		self.y_std = y_std
		self.y_min = np.min(y_seen)

	def get_scores(self, model, x, unseen_indices):
		raise NotImplementedError

	def process_sample(self, scores, y, seen_indices, unseen_indices):
		assert(scores.shape[0] == len(unseen_indices))
		best_index = np.argmax(scores)
		sampled_index = unseen_indices[best_index]
		unseen_indices.pop(best_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)

class ExpectedImprovement(BaseAcquisitionFunction):
	def get_scores(self, model, x, unseen_indices):
		preds, std_devs = model.predict(x[unseen_indices], return_std=True)
		pred_diffs = self.y_min - preds
		z = pred_diffs / (std_devs + 1e-9)
		expect_improves = pred_diffs * norm.cdf(z) + std_devs * norm.pdf(z)
		self.preds = preds
		return expect_improves

class KrigingBelieverEI(ExpectedImprovement, BaseAcquisitionFunction):
	def sample_batch(self, model, x, y, seen_indices, unseen_indices,
	batch_size):
		temp_seen = seen_indices.copy()
		temp_unseen = unseen_indices.copy()
		temp_y = y.copy()
		sampled_indices = list()
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				break
			self.fit_model(model, x, temp_y, temp_seen)
			expect_improves = self.get_scores(model, x, temp_unseen)
			best_index = np.argmax(expect_improves)
			sampled_index = temp_unseen[best_index]
			temp_unseen.pop(best_index)
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = self.y_mean + self.y_std * \
				self.preds[best_index]
			temp_seen.append(sampled_index)
		for sampled_index in sampled_indices:
			unseen_indices.remove(sampled_index)
			if not np.isnan(y[sampled_index]):
				seen_indices.append(sampled_index)
