import numpy as np
from sklearn.linear_model import LogisticRegression

def calculate_opt_features(chi_new_values, n_confs):
	"""
	Calculates features from the chi_new vs r_opt data at the current iteration,
	where chi_new is the amount of a given conformer that would be expected to
	be observed based on its DFT free energy and r_opt is the proportion of the
	total number of conformers that have been optimised.

	Args:
		chi_new_values: list of float of the chi_new values for each conformer
			as they are optimised (missing conformers are not recorded)
		n_confs: int giving the total number of conformers of the molecule

	Returns:
		features: list of float of the features at the current iteration
	"""
	f0, f1, ffrac1, ffrac2, ffrac5 = 0, 0, 0, 0, 0
	p0, p1, pfrac1, pfrac2, pfrac5 = 0, 0, 0, 0, 0
	highest_ropt = 0
	last_forty = int(0.6 * len(chi_new_values))
	for i, chi_new in enumerate(chi_new_values):
		if chi_new == 0.0:
			f0 += 1
			if i >= last_forty:
				p0 += 1
		if chi_new == 1.0:
			f1 += 1
			highest_ropt = i + 1
			if i >= last_forty:
				p1 += 1
		if chi_new <= 0.1:
			ffrac1 += 1
			if i >= last_forty:
				pfrac1 += 1
		if chi_new <= 0.2:
			ffrac2 += 1
			if i >= last_forty:
				pfrac2 += 1
		if chi_new <= 0.5:
			ffrac5 += 1
			if i >= last_forty:
				pfrac5 += 1
	variables = (f0, f1, ffrac1, ffrac2, ffrac5, p0, p1, pfrac1, pfrac2, pfrac5)
	features = [value / len(chi_new_values) for value in variables]
	features += [highest_ropt / n_confs, len(chi_new_values) / n_confs,
		len(chi_new_values)]
	return features

def train_stop_predictor(features, labels):
	"""
	Fit a logistic regression model to a set of optimisation features to predict
	when an optimisation run should be terminated (class label 1) or not (0),
	that is, the lowest energy conformer has been located.

	Args:
		features: 2D numpy array containing the optimisation features
		labels: numpy array containing the stop labels

	Returns:
		stop_predictor: sklearn LogisticRegression model that predicts whether
			an optimisation run should be stopped
	"""
	stop_predictor = LogisticRegression(penalty="l2", max_iter=500)
	stop_predictor.fit(features, labels)
	return stop_predictor

def evaluate_stop_predictions(train_features, train_labels, test_features,
test_min_energies, confidence=0.9):
	"""
	Given sets of features and labels for training and test sets of molecules,
	train a machine learning model to predict whether the conformer optimisation
	should be stopped. Then, calculate the total number of conformers that were
	optimised before stopping, the average proportion of conformers that were
	optimised out of the total number of conformers of each molecule, the number
	of molecules for which a stop was falsely predicted and the average energy
	difference between the lowest energy conformer at the point that the
	optimisation stopped and the true lowest energy conformer, for the test of
	molecules.

	Args:
		train_features: 2D numpy array containing the optimisation features of
			the molecules in the training set
		train_labels: numpy array containing the stop labels for the training
			set molecules
		test_features: 2D numpy array containing the optimisation features of
			the molecules in the test set
		test_min_energies: list of lists of floats of the minimum conformer
			energy found at each sampling iteration for each molecule in the
			test set
		prob_thresh: float giving the probability threshold above which the
			conformer optimisation will be terminated
	"""
	stop_predictor = train_stop_predictor(train_features, train_labels)
	stop_class_idx = np.where(stop_predictor.classes_ == 1)[0][0]
	test_probs = stop_predictor.predict_proba(test_features)[:,stop_class_idx]
	sample_numbers = list()
	proportions = list()
	false_stops = 0
	excess_energies = list()
	last_n_confs = 0
	n_points = 3
	for min_energies in test_min_energies:
		n_confs = len(min_energies)
		predictions = test_probs[last_n_confs:n_confs+last_n_confs]
		stop_indices = np.where(predictions >= confidence)[0]
		if stop_indices.shape[0] > n_points:
			stop_index = stop_indices[0]
			stop_window = np.arange(stop_index, stop_index + n_points,
				dtype=int)
			while not np.all(predictions[stop_window] >= confidence) and \
			stop_window[n_points-1] < predictions.shape[0]:
				stop_window += 1
			stop_index = stop_window[n_points-1]
		else:
			stop_index = predictions.shape[0] - 1
		sample_numbers.append(stop_index + 1)
		proportions.append((stop_index + 1) / n_confs)
		stop_energy = min_energies[stop_index]
		false_stops += int(stop_energy != 0.0)
		excess_energies.append(stop_energy)
		last_n_confs += n_confs
	return sample_numbers, proportions, false_stops, excess_energies
