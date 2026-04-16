#!/usr/bin/env python
import os
import ctypes
import pickle
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.constants import R
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from rdkit import Chem

HARTREE_TO_KCAL = 627.509
HARTREE_TO_JOULES = 2625.5 * 1000.0
TEMPERATURE = 298.15
N_CONV_POINTS = 3

class OptParams():
	def __init__(self):
		self.min_thresh = 0.01
		self.grad_thresh = 0.0001
		self.smoothing = 0.5
		self.batch_size = 1

class JobData():
	def __init__(self, features, ff_energies, qm_energies, seen_indices,
	unseen_indices, kernel_len_scale, init_sampler, archive_filename):
		self.features = features
		self.ff_energies = ff_energies
		self.qm_energies = qm_energies
		self.seen_indices = seen_indices
		self.unseen_indices = unseen_indices
		self.kernel_len_scale = kernel_len_scale
		self.init_sampler = init_sampler
		self.archive_filename = archive_filename
		self.opt_params = OptParams()
		self.score_values = list()
		self.last_update_len = 0

class ForceFieldSampler():
	def __init__(self, ff_energies):
		self.ff_energies = ff_energies
		self.completed = False

	def get_init_sample(self, n_init=5):
		self.ff_sorted_indices = np.argsort(self.ff_energies)
		self.init_indices = [int(self.ff_sorted_indices[i]) \
			for i in range(n_init)]
		self.n_selected = n_init
		return self.init_indices

	def update_init_sample(self, sampled_index, sampled_energy):
		if sampled_index in self.init_indices:
			self.init_indices.remove(sampled_index)
			if np.isnan(sampled_energy):
				if self.n_selected < len(self.ff_energies):
					next_ff_index = self.ff_sorted_indices[self.n_selected]
					self.n_selected += 1
					self.init_indices.append(next_ff_index)
				else:
					print("WARNING: Initial sampler has reached end of force"\
						" field energy list - no more initial samples can be"\
						" taken.")
		if len(self.init_indices) == 0:
			self.completed = True

class KrigingBelieverEI():
	def fit_model(self, model, x, y, seen_indices):
		x_seen = x[seen_indices]
		y_seen = y[seen_indices]
		valid_mask = np.logical_not(np.isnan(y_seen))
		x_train = x_seen[valid_mask]
		y_train = y_seen[valid_mask]
		y_mean, y_std = np.mean(y_train), np.std(y_train)
		y_train = (y_train - y_mean) / y_std
		model.fit(x_train, y_train)
		self.y_mean = y_mean
		self.y_std = y_std
		self.y_min = np.min(y_train)

	def get_scores(self, model, x, unseen_indices):
		preds, std_devs = model.predict(x[unseen_indices], return_std=True)
		pred_diffs = self.y_min - preds
		z = pred_diffs / (std_devs + 1e-9)
		expect_improves = pred_diffs * norm.cdf(z) + std_devs * norm.pdf(z)
		self.preds = preds
		return expect_improves

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
		return sampled_indices

class MainOpts(Enum):
	NEWJOB = 1
	PREVJOB = 2
	CONFIG = 3
	SUGGEST = 4
	UPDATE = 5
	VIEW = 6
	SAVE = 7
	EXIT = 8

class ConfigOpts(Enum):
	MIN = 1
	GRAD = 2
	BATCH = 3
	SMOOTH = 4
	RESET = 5
	EXIT = 6

class ViewOpts(Enum):
	TABLE = 1
	HIST = 2
	PLOT = 3
	LOWEST = 4
	CHI = 5
	EI = 6
	EXIT = 7


def check_ready():
	global job_data
	if job_data is None:
		return False
	return True

def validate_filename(filename):
	if not os.path.exists(filename):
		print("ERROR: File '%s' does not exist." % filename)
		return False
	if not os.access(filename, os.R_OK):
		print("ERROR: Cannot access file '%s' - permission denied." % filename)
		return False
	return True

def start_new_job():
	global job_data
	print("Starting new minimisation.")
	if check_ready():
		print("Saving current data to archive '%s' before starting new..." % \
			job_data.archive_filename)
		save_data()
	while True:
		sdf_filename = input("Enter .sdf file of conformers: ")
		if sdf_filename.lower() == "quit" or sdf_filename.lower() == "exit":
			return
		if not validate_filename(sdf_filename):
			continue
		try:
			mols = list(Chem.SDMolSupplier(sdf_filename, sanitize=False,
				removeHs=False))
		except:
			print("ERROR: Could not read conformers from '%s'." % sdf_filename)
			continue
		ff_filename = input("Enter .npy file of low level conformer energies: ")
		if ff_filename.lower() == "quit" or ff_filename.lower() == "exit":
			return
		if not validate_filename(ff_filename):
			continue
		try:
			ff_energies = np.load(ff_filename)
		except:
			print("ERROR: Could not read energies from '%s'." % ff_filename)
		if len(mols) != ff_energies.shape[0]:
			print("ERROR: Different numbers of conformers and low level"\
				" energies.")
			continue
		break
	archive_filename = os.path.basename(sdf_filename).replace(".sdf",
		 ".cnfmin.pkl")
	if not os.access(".", os.W_OK):
		print("ERROR: Cannot write file '%s' to current working directory -"\
			" permission denied." % archive_filename)
		return
	if not validate_filename("./libgeometry.so"):
		return
	lib = ctypes.CDLL("./libgeometry.so")
	lib.rbf_maximise_variance.restype = ctypes.c_double
	print("Calculating inverse interatomic distance features...")
	features = list()
	n_atoms = mols[0].GetNumAtoms()
	n_dists = (n_atoms * (n_atoms - 1)) // 2
	n_atoms_inp = ctypes.c_size_t(n_atoms)
	n_dists_inp = ctypes.c_size_t(n_dists)
	for mol in mols:
		conf = mol.GetConformer()
		coords = conf.GetPositions()
		values = np.empty(n_dists)
		coords_inp = np.ctypeslib.as_ctypes(coords)
		values_inp = np.ctypeslib.as_ctypes(values)
		lib.calculate_inv_dists(coords_inp, n_atoms_inp, values_inp,
			n_dists_inp)
		features.append(values)
	features = np.array(features)
	features = StandardScaler().fit_transform(features)
	pca = PCA(n_components=0.99, svd_solver="full")
	print("Compressing features...")
	features = pca.fit_transform(features)
	features = features.copy().reshape(features.shape, order="C")
	print("Calculating RBF kernel length scale...")
	features_inp = np.ctypeslib.as_ctypes(features)
	feature_rows_inp = ctypes.c_size_t(features.shape[0])
	feature_cols_inp = ctypes.c_size_t(features.shape[1])
	ubound_inp = ctypes.c_int(3)
	lbound_inp = ctypes.c_int(-6)
	kernel_len_scale = lib.rbf_maximise_variance(features_inp, feature_rows_inp,
		feature_cols_inp, ubound_inp, lbound_inp)
	if kernel_len_scale <= 0.0:
		print("ERROR: Failed to calculate RBF kernel length scale.")
		return
	print("Determining initial sample...")
	init_sampler = ForceFieldSampler(ff_energies)
	init_indices = init_sampler.get_init_sample()
	print("The following conformers are recommended for calculation: %s" % \
		" ".join(str(init_index + 1) for init_index in init_indices))
	seen_indices = list()
	unseen_indices = list(range(features.shape[0]))
	qm_energies = np.zeros_like(ff_energies)
	job_data = JobData(features, ff_energies, qm_energies, seen_indices,
		unseen_indices, kernel_len_scale, init_sampler, archive_filename)
	print("Data will be saved to the archive file: '%s'." % archive_filename)

def load_prev_job():
	global job_data
	print("Loading previous minimisation.")
	if check_ready():
		print("Saving current data to archive '%s' before loading..." % \
			job_data.archive_filename)
		save_data()
	while True:
		input_archive_name = input("Enter the archive (.cnfmin.pkl) file to"\
			" load: ")
		if input_archive_name.lower() == "quit" or \
		input_archive_name.lower() == "exit":
			return
		if not validate_filename(input_archive_name):
			continue
		try:
			archive_file = open(input_archive_name, "rb")
			job_data = pickle.load(archive_file)
		except:
			print("ERROR: Could not read data from '%s'." % input_archive_name)
			continue
		break

def select_config_option():
	global job_data
	opt_params = job_data.opt_params
	min_thresh = opt_params.min_thresh
	grad_thresh = opt_params.grad_thresh
	smoothing = opt_params.smoothing
	batch_size = opt_params.batch_size
	option_text = "\nSelect option (type quit/exit to go to previous menu):\n"\
		"1. Set minimum EI proportion. Currently: %f, default: 0.01\n"\
		"2. Set EI gradient threshold. Currently: %f, default: 0.0001\n"\
		"3. Set batch size. Currently: %d, default: 1\n"\
		"4. Set smoothing parameter. Currently: %f, default: 0.5\n"\
		"5. Reset to defaults.\n"\
		"6. Return to previous menu.\n" % (min_thresh, grad_thresh, batch_size,
		smoothing)
	while True:
		selection = input(option_text)
		if selection.lower() == "quit" or selection.lower() == "exit":
			return ConfigOpts.EXIT.value
		try:
			selection = int(selection)
		except ValueError:
			if selection != "":
				print("ERROR: Input '%s' is invalid." % selection)
			continue
		if selection < ConfigOpts.MIN.value or \
		selection > ConfigOpts.EXIT.value:
			print("ERROR: Input '%d' is invalid." % selection)
			continue
		else:
			break
	return selection

def configure_parameters():
	global job_data
	print("Configuring optimisation settings.")
	if not check_ready():
		print("WARNING: A minimisation job has not been created or loaded."\
			" Cannot update settings.")
		return
	while True:
		selection = select_config_option()
		if selection == ConfigOpts.MIN.value:
			print("For convergence to be considered, the last three mean EI"\
				" values must be below this proportion of the maximum mean EI"\
				" value found so far (default 1%).")
			new_min_thresh = input("Enter new value: ")
			try:
				new_min_thresh = float(new_min_thresh)
			except ValueError:
				print("ERROR: Input '%s' is invalid." % new_min_thresh)
				continue
			if np.isnan(new_min_thresh) or np.isinf(new_min_thresh) or \
			new_min_thresh <= 0.0:
				print("ERROR: Input '%f' is invalid." % new_min_thresh)
				continue
			job_data.opt_params.min_thresh = new_min_thresh
		elif selection == ConfigOpts.GRAD.value:
			print("This sets the value below which the gradient of the mean EI"\
				" curve must be for convergence to be considered (default"\
				" 10^-4).")
			new_grad_thresh = input("Enter new value: ")
			try:
				new_grad_thresh = float(new_grad_thresh)
			except ValueError:
				print("ERROR: Input '%s' is invalid." % new_grad_thresh)
				continue
			if np.isnan(new_grad_thresh) or np.isinf(new_grad_thresh) or \
			new_grad_thresh <= 0.0:
				print("ERROR: Input '%f' is invalid." % new_grad_thresh)
				continue
			job_data.opt_params.grad_thresh = new_grad_thresh
		elif selection == ConfigOpts.SMOOTH.value:
			print("This sets the exponential smoothing parameter which"\
				" controls the extend to which previous values of the mean EI"\
				" values influence the current one (default 0.5).\n"\
				"It is not recommended to change this value after beginning"\
				" an optimization job.")
			new_smoothing = input("Enter new value: ")
			try:
				new_smoothing = float(new_smoothing)
			except ValueError:
				print("ERROR: Input '%s' is invalid." % new_smoothing)
				continue
			if np.isnan(new_smoothing) or np.isinf(new_smoothing) or \
			new_smoothing <= 0.0 or new_smoothing > 1.0:
				print("ERROR: Input '%f' is invalid." % new_smoothing)
				continue
			job_data.opt_params.smoothing = new_smoothing
		elif selection == ConfigOpts.BATCH.value:
			print("This sets the batch size, i.e. the number of conformers"\
				" that will be suggested at a time, using the Kriging Believer"\
				" method to select more than one conformer (default 1).")
			new_batch_size = input("Enter new value: ")
			try:
				new_batch_size = int(new_batch_size)
			except ValueError:
				print("ERROR: Input '%s' is invalid." % new_batch_size)
				continue
			if new_batch_size <= 0 or \
			new_batch_size > len(job_data.unseen_indices):
				print("ERROR: A batch size of %d is invalid." % new_batch_size)
				continue
			job_data.opt_params.batch_size = new_batch_size
		elif selection == ConfigOpts.RESET.value:
			print("Resetting to defaults...")
			job_data.opt_params = OptParams()
		elif selection == ConfigOpts.EXIT.value:
			return

def check_convergence(score_values, opt_params):
	if len(score_values) < N_CONV_POINTS:
		return False
	max_val = max(score_values)
	prev_vals = score_values[-N_CONV_POINTS:]
	print("Last mean EI values (%.2f%% of max = %.6f): %s" % \
		(100.0 * opt_params.min_thresh, opt_params.min_thresh * max_val,
		" ".join("%.6f" % val for val in prev_vals)))
	check1 = all(value < opt_params.min_thresh * max_val for value in prev_vals)
	grad_est = (prev_vals[0] - 4.0 * prev_vals[1] + 3.0 * prev_vals[2]) / 2.0
	print("Estimated gradient: %.6f" % grad_est)
	check2 = abs(grad_est) < opt_params.grad_thresh
	if check1 and check2:
		return True
	return False

def suggest_next():
	global job_data
	print("Suggesting next conformers.")
	if not check_ready():
		print("WARNING: A minimisation job has not been created or loaded."\
			" No data are available to make suggestions upon.")
		return
	if not job_data.init_sampler.completed:
		print("The initial sample has not yet been completed. The following"\
			" conformers are recommended: %s" % \
			" ".join(str(init_index + 1) for init_index in \
			job_data.init_sampler.init_indices))
	else:
		print("Building model...")
		kernel = RBF(length_scale=job_data.kernel_len_scale,
			length_scale_bounds="fixed")
		model = GaussianProcessRegressor(kernel=kernel, optimizer=None,
			normalize_y=True)
		acq_func = KrigingBelieverEI()
		print("Running acquisition...")
		if len(job_data.seen_indices) >= job_data.last_update_len + \
		job_data.opt_params.batch_size:
			acq_func.fit_model(model, job_data.features, job_data.qm_energies,
				job_data.seen_indices)
			acq_scores = acq_func.get_scores(model, job_data.features,
				job_data.unseen_indices)
			score = float(np.mean(acq_scores))
			if len(job_data.score_values) > 0:
				smoothing = job_data.opt_params.smoothing
				new_score = smoothing * score + \
					(1.0 - smoothing) * job_data.score_values[-1]
			else:
				new_score = score
			job_data.score_values.append(new_score)
			job_data.last_update_len = len(job_data.seen_indices)
		if check_convergence(job_data.score_values, job_data.opt_params):
			print("HOORAY the convergence criteria have been met!\nIt is"\
				" possible that the lowest energy conformer has been found.")
		selected_indices = acq_func.sample_batch(model, job_data.features,
			job_data.qm_energies, job_data.seen_indices,
			job_data.unseen_indices, job_data.opt_params.batch_size)
		print("The following conformers are recommended: %s" % \
			" ".join(str(index + 1) for index in selected_indices))

def update_data():
	global job_data
	print("Updating energy data.")
	if not check_ready():
		print("WARNING: A minimisation job has not been created or loaded."\
			" No data are available to update.")
		return
	while True:
		conf_number = input("Enter the conformer number to be updated: ")
		if conf_number.lower() == "quit" or conf_number.lower() == "exit":
			return
		try:
			conf_number = int(conf_number)
		except ValueError:
			print("ERROR: Input '%s' is invalid." % conf_number)
			continue
		if conf_number < 1 or conf_number > len(job_data.ff_energies):
			print("ERROR: Input '%d' is outside the range of the number of"\
				" conformers (%d)." % (conf_number, len(job_data.ff_energies)))
			continue
		conf_index = conf_number - 1
		conf_energy = input("Conformer %d energy: ('nan' if unavailable or"\
			" 'del' to add back): " % conf_number)
		if conf_energy.lower() == "quit" or conf_energy.lower() == "exit":
			return
		elif conf_energy.lower() == "del":
			if conf_index in job_data.seen_indices:
				job_data.seen_indices.remove(conf_index)
				job_data.unseen_indices.append(conf_index)
				qm_energies[conf_index] = 0.0
			else:
				print("WARNING: Conformer %d has not yet been optimised. Not"\
					" moving back to the search space." % conf_number)
			return
		try:
			conf_energy = float(conf_energy)
		except ValueError:
			print("ERROR: Input '%s' is invalid." % conf_energy)
			continue
		if np.isinf(conf_energy):
			print("ERROR: Cannot have infinite energy.")
			continue
		break
	if conf_index not in job_data.seen_indices:
		if len(job_data.seen_indices) > 0 and \
		conf_energy < np.nanmin(job_data.qm_energies[job_data.seen_indices]):
			print("HOORAY a new lowest energy conformer has been found!")
		job_data.seen_indices.append(conf_index)
		job_data.qm_energies[conf_index] = conf_energy
	else:
		overwrite_input = input("WARNING: Conformer %d has already been"\
			" calculated with energy %.6f. Do you want to overwrite? (Y/n) " % \
			(conf_number, job_data.qm_energies[conf_index]))
		if overwrite_input == "" or overwrite_input[0].lower() == 'y':
			if conf_energy < \
			np.nanmin(job_data.qm_energies[job_data.seen_indices]):
				print("HOORAY a new lowest energy conformer has been found!")
			job_data.qm_energies[conf_index] = conf_energy
		else:
			return
	if conf_index in job_data.unseen_indices:
		job_data.unseen_indices.remove(conf_index)
	if not job_data.init_sampler.completed:
		job_data.init_sampler.update_init_sample(conf_index, conf_energy)

def select_view_option():
	option_text = "\nSelect option (type quit/exit to go to previous menu):\n"\
		"1. Print currently calculated conformers and their energies.\n"\
		"2. Plot histogram of conformer energies.\n"\
		"3. Plot sampled energies in order.\n"\
		"4. Plot lowest sampled energies.\n"\
		"5. Plot exp(-(Enew - Emin)/RT) against fraction optimised.\n"\
		"6. Plot mean EI scores at each iteration.\n"\
		"7. Return to previous menu.\n"
	while True:
		selection = input(option_text)
		if selection.lower() == "quit" or selection.lower() == "exit":
			return ViewOpts.EXIT.value
		try:
			selection = int(selection)
		except ValueError:
			if selection != "":
				print("ERROR: Input '%s' is invalid." % selection)
			continue
		if selection < ViewOpts.TABLE.value or \
		selection > ViewOpts.EXIT.value:
			print("ERROR Input '%d' is invalid." % selection)
			continue
		else:
			break
	return selection

def view_results():
	global job_data
	print("Viewing results so far.")
	if not check_ready():
		print("WARNING: A minimisation job has not been created or loaded."\
			" No data are available to view.")
		return
	if len(job_data.seen_indices) == 0:
		print("No energies have been collected yet.")
		return
	seen_indices = job_data.seen_indices
	qm_energies = job_data.qm_energies
	valid_mask = np.logical_not(np.isnan(qm_energies[seen_indices]))
	energies = qm_energies[seen_indices][valid_mask]
	while True:
		selection = select_view_option()
		if selection == ViewOpts.TABLE.value:
			min_energy = np.min(energies)
			print("Conformer | Energy (Hartree) | Rel. Energy (kcal/mol)")
			print("-----------------------------------------------------")
			for index in job_data.seen_indices:
				print("%9d | %16.6f | %16.4f" % (index + 1, qm_energies[index],
					HARTREE_TO_KCAL * (qm_energies[index] - min_energy)))
			print("-----------------------------------------------------")
		elif selection == ViewOpts.HIST.value:
			if len(energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			# Use Freedman-Diaconis rule to calculate number of bins
			low_quart = np.percentile(energies, 25, method="higher")
			high_quart = np.percentile(energies, 75, method="lower")
			inter_range = high_quart - low_quart
			bin_width = 2.0 * inter_range / len(energies)**(1.0/3.0)
			if bin_width == 0.0:
				print("WARNING: Not enough data for a histogram.")
				continue
			bins = np.arange(np.min(energies), np.max(energies) + bin_width,
				bin_width)
			print("Number of bins set to %d." % len(bins))
			plt.hist(energies, bins=bins)
			plt.xlabel("Energy / Hartree", fontsize=14)
			plt.title("Conformer Energies", fontsize=14)
			plt.show()
		elif selection == ViewOpts.PLOT.value:
			if len(energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			plt.plot(np.arange(len(energies)) + 1, energies, "o-")
			plt.xlabel("Sample Number (ignoring NaN points)", fontsize=14)
			plt.ylabel("Energy / Hartree", fontsize=14)
			plt.title("Sampled Energies", fontsize=14)
			plt.show()
		elif selection == ViewOpts.LOWEST.value:
			if len(energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			min_energy = energies[0]
			min_energies = list()
			for energy in energies:
				min_energy = min(min_energy, energy)
				min_energies.append(min_energy)
			plt.plot(np.arange(len(min_energies)) + 1, energies, "o-")
			plt.xlabel("Sample Number (ignoring NaN points)", fontsize=14)
			plt.ylabel("Min Energy / Hartree", fontsize=14)
			plt.title("Lowest Sampled Energies", fontsize=14)
			plt.show()
		elif selection == ViewOpts.CHI.value:
			print("Plot of chi_new against r_opt from C. C. Lam and J. M."\
				" Goodman:\nJ. Chem. Inf. Model., 2023, 63, 4364-4375, DOI:"\
				" 10.1021/acs.jcim.3c00649")
			ropt = list()
			chi_new = list()
			for i, index in enumerate(seen_indices):
				ropt.append((i + 1) / job_data.qm_energies.shape[0])
				energy = qm_energies[index]
				if not np.isnan(energy):
					min_energy = np.nanmin(energies[:i+1])
					dg_new = HARTREE_TO_JOULES * (energy - min_energy) \
						/ (R * TEMPERATURE)
					chi_new.append(np.exp(-dg_new))
				else:
					chi_new.append(0.0)
			plt.plot(ropt, chi_new)
			plt.xlim(-0.02, 1.05)
			plt.xlabel("$r_{opt}$", fontsize=14)
			plt.ylabel("exp(-($E_{new} - E_{min}$)/RT)", fontsize=14)
			plt.show()
		elif selection == ViewOpts.EI.value:
			score_values = job_data.score_values
			if len(score_values) == 0:
				print("WARNING: No acquisition function scores have been"\
					" recorded.")
				continue
			plt.plot(np.arange(len(score_values)) + 1, score_values, "o-")
			plt.xlabel("Sample Number", fontsize=14)
			plt.title("Mean EI Scores", fontsize=14)
			plt.show()
		elif selection == ViewOpts.EXIT.value:
			return

def save_data():
	global job_data
	if check_ready():
		print("Saving data to archive '%s'..." % job_data.archive_filename)
		archive_file = open(job_data.archive_filename, "wb")
		pickle.dump(job_data, archive_file)
	else:
		print("No data has been created or loaded. Not saving.")

def select_main_option():
	option_text = "\nSelect option (type quit/exit to exit the program):\n"\
		"1. Start a new conformer energy minimisation.\n"\
		"2. Load and continue a previous minimisation.\n"\
		"3. Configure optimisation settings.\n"\
		"4. Suggest next conformers.\n"\
		"5. Update calculated energy.\n"\
		"6. View results so far.\n"\
		"7. Save results and data.\n"\
		"8. Save and exit.\n"
	while True:
		selection = input(option_text)
		if selection.lower() == "quit" or selection.lower() == "exit":
			return MainOpts.EXIT.value
		try:
			selection = int(selection)
		except ValueError:
			if selection != "":
				print("ERROR: Input '%s' is invalid." % selection)
			continue
		if selection < MainOpts.NEWJOB.value or selection > MainOpts.EXIT.value:
			print("ERROR: Input '%d' is invalid." % selection)
			continue
		else:
			break
	return selection

def main():
	while True:
		selection = select_main_option()
		if selection == MainOpts.NEWJOB.value:
			start_new_job()
		elif selection == MainOpts.PREVJOB.value:
			load_prev_job()
		elif selection == MainOpts.CONFIG.value:
			configure_parameters()
		elif selection == MainOpts.SUGGEST.value:
			suggest_next()
		elif selection == MainOpts.UPDATE.value:
			update_data()
		elif selection == MainOpts.VIEW.value:
			view_results()
		elif selection == MainOpts.SAVE.value:
			save_data()
		elif selection == MainOpts.EXIT.value:
			save_data()
			print("Exiting...")
			return

if __name__ == "__main__":
	job_data = None
	main()
