import numpy as np
from scipy.stats import norm

class BaseAcquisitionFunction():
	def __init__(self):
		self.surrogate = None

	def fit_predict(self, x, y, seen_indices, unseen_indices, model, refit):
		x_seen = x[seen_indices]
		y_seen = y[seen_indices]
		y_mean, y_std = y_seen.mean(), y_seen.std()
		y_seen = (y_seen - y_mean) / y_std
		if refit:
			self.surrogate = model.fit(x_seen, y_seen)
		elif self.surrogate is None:
			print("ERROR: Surrogate model is not fit.")
			return None
		preds, std_devs = self.surrogate.predict(x[unseen_indices], return_std=True)
		return preds, std_devs, y_seen.min()

	def select_next(self, x, y, seen_indices, unseen_indices, model, refit=True):
		raise NotImplementedError

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices, model, n_samples):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		else:
			return n_samples
		n_samples += 1
		while np.isnan(y[sampled_index]) and len(unseen_indices) > 0:
			sampled_index = self.select_next(x, y, seen_indices, unseen_indices, model, False)
			unseen_indices.remove(sampled_index)
			n_samples += 1
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)
		return n_samples

class Exploitation(BaseAcquisitionFunction):
	def __init__(self):
		super(Exploitation, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, refit=True):
		preds, std_devs, y_min = self.fit_predict(x, y, seen_indices, unseen_indices, model, refit)
		minimum_indices = np.where(preds == preds.min())[0]
		# Break ties randomly
		minimum_index = np.random.choice(minimum_indices)
		return unseen_indices[minimum_index]

class ProbabilityImprovement(BaseAcquisitionFunction):
	def __init__(self):
		super(ProbabilityImprovement, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, refit=True):
		preds, std_devs, y_min = self.fit_predict(x, y, seen_indices, unseen_indices, model, refit)
		z = (y_min - preds) / (std_devs + 1e-9)
		prob_improve = norm.cdf(z)
		# Break ties randomly
		best_indices = np.where(prob_improve == prob_improve.max())[0]
		best_index = np.random.choice(best_indices)
		return unseen_indices[best_index]

class ExpectedImprovement(BaseAcquisitionFunction):
	def __init__(self):
		super(ExpectedImprovement, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, refit=True):
		preds, std_devs, y_min = self.fit_predict(x, y, seen_indices, unseen_indices, model, refit)
		pred_diffs = y_min - preds
		z = pred_diffs / (std_devs + 1e-9)
		expect_improve = pred_diffs * norm.cdf(z) + std_devs * norm.pdf(z)
		# Break ties randomly
		best_indices = np.where(expect_improve == expect_improve.max())[0]
		best_index = np.random.choice(best_indices)
		return unseen_indices[best_index]

class LowerConfidenceBound(BaseAcquisitionFunction):
	def __init__(self):
		super(LowerConfidenceBound, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, refit=True):
		preds, std_devs, y_min = self.fit_predict(x, y, seen_indices, unseen_indices, model, refit)
		lcb = preds - std_devs
		# Break ties randomly
		best_indices = np.where(lcb == lcb.min())[0]
		best_index = np.random.choice(best_indices)
		return unseen_indices[best_index]

class BatchedPI(BaseAcquisitionFunction):
	def __init__(self):
		super(BatchedPI, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, batch_size, refit=True):
		preds, std_devs, y_min = self.fit_predict(x, y, seen_indices, unseen_indices, model, refit)
		z = (y_min - preds) / (std_devs + 1e-9)
		prob_improve = norm.cdf(z)
		sorted_indices = np.argsort(prob_improve)[::-1]
		sampled_indices = list()
		for i in range(batch_size):
			if i >= len(sorted_indices):
				i -= 1
				break
			sampled_index = unseen_indices[sorted_indices[i]]
			sampled_indices.append(sampled_index)
		for sampled_index in sampled_indices:
			self.process_sample(sampled_index, x, y, seen_indices, unseen_indices, model)
		return i + 1

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices, model):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)

class KrigingBelieverPI(BaseAcquisitionFunction):
	def __init__(self):
		super(KrigingBelieverPI, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, batch_size, refit=True):
		temp_y = y.copy()
		sampled_indices = list()
		temp_unseen = unseen_indices.copy()
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				i -= 1
				break
			y_mean = np.mean(temp_y[seen_indices+sampled_indices])
			y_std = np.std(temp_y[seen_indices+sampled_indices])
			preds, std_devs, y_min = self.fit_predict(x, temp_y, seen_indices + sampled_indices, temp_unseen, model, refit)
			z = (y_min - preds) / (std_devs + 1e-9)
			prob_improve = norm.cdf(z)
			max_index = np.argmax(prob_improve)
			sampled_index = temp_unseen[max_index]
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = y_mean + y_std * preds[max_index]
			temp_unseen.pop(max_index)
		for sampled_index in sampled_indices:
			self.process_sample(sampled_index, x, y, seen_indices, unseen_indices)
		return i + 1

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)

class MinConstantLiarPI(BaseAcquisitionFunction):
	def __init__(self):
		super(MinConstantLiarPI, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, batch_size, refit=True):
		temp_y = y.copy()
		y_min_sub = np.min(y[seen_indices])
		sampled_indices = list()
		temp_unseen = unseen_indices.copy()
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				i -= 1
				break
			y_mean = np.mean(temp_y[seen_indices+sampled_indices])
			y_std = np.std(temp_y[seen_indices+sampled_indices])
			preds, std_devs, y_min = self.fit_predict(x, temp_y, seen_indices + sampled_indices, temp_unseen, model, refit)
			z = (y_min - preds) / (std_devs + 1e-9)
			prob_improve = norm.cdf(z)
			max_index = np.argmax(prob_improve)
			sampled_index = temp_unseen[max_index]
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = y_min_sub
			temp_unseen.pop(max_index)
		for sampled_index in sampled_indices:
			self.process_sample(sampled_index, x, y, seen_indices, unseen_indices)
		return i + 1

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)

class MaxConstantLiarPI(BaseAcquisitionFunction):
	def __init__(self):
		super(MaxConstantLiarPI, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, batch_size, refit=True):
		temp_y = y.copy()
		y_max_sub = np.max(y[seen_indices])
		sampled_indices = list()
		temp_unseen = unseen_indices.copy()
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				i -= 1
				break
			y_mean = np.mean(temp_y[seen_indices+sampled_indices])
			y_std = np.std(temp_y[seen_indices+sampled_indices])
			preds, std_devs, y_min = self.fit_predict(x, temp_y, seen_indices + sampled_indices, temp_unseen, model, refit)
			z = (y_min - preds) / (std_devs + 1e-9)
			prob_improve = norm.cdf(z)
			max_index = np.argmax(prob_improve)
			sampled_index = temp_unseen[max_index]
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = y_max_sub
			temp_unseen.pop(max_index)
		for sampled_index in sampled_indices:
			self.process_sample(sampled_index, x, y, seen_indices, unseen_indices)
		return i + 1

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)

class MeanConstantLiarPI(BaseAcquisitionFunction):
	def __init__(self):
		super(MeanConstantLiarPI, self).__init__()

	def select_next(self, x, y, seen_indices, unseen_indices, model, batch_size, refit=True):
		temp_y = y.copy()
		y_mean_sub = np.mean(y[seen_indices])
		sampled_indices = list()
		temp_unseen = unseen_indices.copy()
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				i -= 1
				break
			y_mean = np.mean(temp_y[seen_indices+sampled_indices])
			y_std = np.std(temp_y[seen_indices+sampled_indices])
			preds, std_devs, y_min = self.fit_predict(x, temp_y, seen_indices + sampled_indices, temp_unseen, model, refit)
			z = (y_min - preds) / (std_devs + 1e-9)
			prob_improve = norm.cdf(z)
			max_index = np.argmax(prob_improve)
			sampled_index = temp_unseen[max_index]
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = y_mean_sub
			temp_unseen.pop(max_index)
		for sampled_index in sampled_indices:
			self.process_sample(sampled_index, x, y, seen_indices, unseen_indices)
		return i + 1

	def process_sample(self, sampled_index, x, y, seen_indices, unseen_indices):
		if len(unseen_indices) > 0:
			unseen_indices.remove(sampled_index)
		if not np.isnan(y[sampled_index]):
			seen_indices.append(sampled_index)
