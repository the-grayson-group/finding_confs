#!/usr/bin/env python
import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from scipy.constants import R
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from rdkit.Chem import SDMolSupplier

# global data
features = None
ff_energies = None
qm_energies = None
seen_indices = None
unseen_indices = None
init_indices = None
kernel_len_scale = None
archive_filename = None
init_sampler = None

class ForceFieldSampler():
	def __init__(self, ff, selected_indices):
		self.ff = ff
		self.selected_indices = selected_indices

	def get_init_sample(self, n_init=5):
		self.ff_sorted_indices = np.argsort(self.ff)
		self.selected_indices = [int(self.ff_sorted_indices[i]) for i in range(n_init)]
		self.n_selected = n_init
		return self.selected_indices

	def update_init_sample(self, sampled_index, sampled_energy):
		if sampled_index in self.selected_indices:
			if np.isnan(sampled_energy):
				if self.n_selected < len(self.ff):
					next_ff_index = self.ff_sorted_indices[self.n_selected]
					self.n_selected += 1
					chosen_init = self.selected_indices.index(sampled_index)
					self.selected_indices[chosen_init] = next_ff_index
				else:
					print("WARNING: Initial sampler has reached end of force field energy list.")
					self.selected_indices.remove(sampled_index)
			else:
				self.selected_indices.remove(sampled_index)
		return self.selected_indices

def check_ready():
	global features
	global ff_energies
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global kernel_len_scale
	global archive_filename
	global init_sampler
	if features is None or ff_energies is None or qm_energies is None or \
		seen_indices is None or unseen_indices is None or \
		init_indices is None or kernel_len_scale is None \
		or init_sampler is None or archive_filename is None:
		return False
	return True

def select_main_option():
	option_text = "\nSelect option (type quit/exit to exit the program):\n" \
	"1. Start a new conformer energy minimisation.\n" \
	"2. Load and continue a previous minimisation.\n" \
	"3. Update calculated conformer energy.\n" \
	"4. Suggest next conformer.\n" \
	"5. View results so far.\n" \
	"6. Save results and data.\n" \
	"7. Save and exit.\n"
	while True:
		selection = input(option_text)
		if selection.lower() == "quit" or selection.lower() == "exit":
			return 7
		try:
			selection = int(selection)
		except ValueError:
			print("ERROR: Input '%s' is invalid." % selection)
			continue
		if selection < 1 or selection > 7:
			print("ERROR: Input '%d' is invalid." % selection)
			continue
		else:
			break
	return selection

def start_new_job():
	global features
	global ff_energies
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global kernel_len_scale
	global archive_filename
	global init_sampler
	print("Starting new minimisation.")
	if check_ready():
		print("Saving current data to archive '%s' before starting new..." % archive_filename)
		save_data()
	while True:
		sdf_filename = input("Enter .sdf file of conformers: ")
		if sdf_filename.lower() == "quit" or sdf_filename.lower() == "exit":
			return
		try:
			mols = [mol for mol in SDMolSupplier(sdf_filename, sanitize=False)]
		except:
			print("ERROR: Could not read conformers from '%s'." % sdf_filename)
			continue
		ff_filename = input("Enter .npy file of force field energies: ")
		if ff_filename.lower() == "quit" or ff_filename.lower() == "exit":
			return
		try:
			ff_energies = np.load(ff_filename)
		except PermissionError:
			print("ERROR: Could not open '%s' - permission denied." % ff_filename)
			continue
		except FileNotFoundError:
			print("ERROR: '%s' - no such file or directory." % ff_filename)
			continue
		except:
			print("ERROR: Could not read energies from '%s'." % ff_filename)
			continue
		if len(mols) != ff_energies.shape[0]:
			print("ERROR: Different number of conformers and force field energies.\n")
			continue
		break
	archive_filename = os.path.basename(sdf_filename).replace(".sdf", ".cnfmin.npz")
	lib = ctypes.CDLL("./libgeometry.so")
	lib.rbf_maximise_variance.restype = ctypes.c_double
	print("Calculating features...")
	features = list()
	for i, mol in enumerate(mols):
		conf = mol.GetConformer()
		coords = conf.GetPositions()
		atoms = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
		n_nonh = np.where(atoms != 1)[0].shape[0]
		n_dists = (n_nonh * (n_nonh - 1)) // 2
		values = np.empty(n_dists)
		lib.calculate_inv_dists(np.ctypeslib.as_ctypes(coords), np.ctypeslib.as_ctypes(atoms), ctypes.c_size_t(atoms.shape[0]), np.ctypeslib.as_ctypes(values), ctypes.c_size_t(n_dists))
		features.append(values)
	features = np.array(features)
	print("Removing low variance features...")
	features = VarianceThreshold(0.0001).fit_transform(features)
	print("Standardising features...")
	features = StandardScaler().fit_transform(features)
	print("Performing PCA on features...")
	n_dims = min(30, features.shape[0], features.shape[1])
	pca = PCA(n_components=n_dims, random_state=5)
	print("Reducing dimensionality to %d..." % n_dims)
	features = pca.fit_transform(features)
	print("Adding force field energies as a feature...")
	features = np.concatenate((features, ff_energies.reshape(-1, 1)), axis=1)
	print("Re-standardising features...")
	features = StandardScaler().fit_transform(features)
	features = features.copy().reshape(features.shape, order="C")
	print("Calculating kernel length scale...")
	kernel_len_scale = lib.rbf_maximise_variance(np.ctypeslib.as_ctypes(features), ctypes.c_size_t(features.shape[0]), ctypes.c_size_t(features.shape[1]), ctypes.c_int(3), ctypes.c_int(-6))
	if kernel_len_scale < 0.0:
		print("ERROR: Failed to calculate kernel bandwidth.")
		features = None
		ff_energies = None
		kernel_len_scale = None
		archive_filename = None
		return
	print("Determining initial sample...")
	init_sampler = ForceFieldSampler(ff_energies, None)
	init_indices = init_sampler.get_init_sample()
	print("The following conformers are recommended for calculation: %s" % " ".join(str(init_index + 1) for init_index in init_indices))
	seen_indices = list()
	unseen_indices = [i for i in range(len(ff_energies))]
	qm_energies = np.zeros_like(ff_energies)
	print("Data will be saved to the archive file: '%s'." % archive_filename)

def load_prev_job():
	global features
	global ff_energies
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global kernel_len_scale
	global archive_filename
	global init_sampler
	print("Loading previous minimisation.")
	if check_ready():
		print("Saving current data to archive '%s' before loading..." % archive_filename)
		save_data()
	while True:
		input_archive_name = input("Enter the archive name to load: ")
		if input_archive_name.lower() == "quit" or input_archive_name.lower() == "exit":
			return
		try:
			data = np.load(input_archive_name)
			features = data["features"]
			ff_energies = data["ff_energies"]
			qm_energies = data["qm_energies"]
			seen_indices = [int(i) for i in data["seen_indices"]]
			unseen_indices = [int(i) for i in data["unseen_indices"]]
			init_indices = [int(i) for i in data["init_indices"]]
			kernel_len_scale = data["kernel_len_scale"]
			archive_filename = input_archive_name
			init_sampler = ForceFieldSampler(ff_energies, init_indices)
			print("Data successfully loaded from '%s'." % archive_filename)
			break
		except PermissionError:
			print("ERROR: Could not open '%s' - permission denied." % input_archive_name)
			continue
		except FileNotFoundError:
			print("ERROR: '%s' - no such file or directory." % input_archive_name)
			continue
		except:
			print("ERROR: Could not read energies from '%s'." % input_archive_name)
			continue

def update_data():
	global ff_energies
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global init_sampler
	print("Updating energy data.")
	if not check_ready():
		print("WARNING: A minimisation has not been created or loaded. No data are available to update.")
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
		if conf_number < 1 or conf_number > len(ff_energies):
			print("ERROR: Input '%d' is outside of the range of the number of conformers (%d)." % (conf_number, len(ff_energies)))
			continue
		conf_energy = input("Energy of conformer %d: ('nan' if unavailable or 'del' to add back to search): " % conf_number)
		if conf_energy.lower() == "quit" or conf_energy.lower() == "exit":
			return
		elif conf_energy.lower() == "del":
			if conf_number - 1 in seen_indices:
				seen_indices.remove(conf_number - 1)
				unseen_indices.append(conf_number - 1)
				qm_energies[conf_number-1] = 0.0
			else:
				print("WARNING: Conformer %d is not in the current set of calculated conformers. Not moving back to the search space." % conf_number)
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
	if conf_number - 1 not in seen_indices:
		seen_indices.append(conf_number - 1)
		if conf_energy < np.nanmin(qm_energies[seen_indices]):
			print("HOORAY a new lowest energy conformer has been found.")
		qm_energies[conf_number-1] = conf_energy
	else:
		overwrite_input = input("WARNING: Conformer %d has already been calculated with energy %.7f. Do you want to overwrite? (Y/n) " % (conf_number, qm_energies[conf_number-1]))
		if overwrite_input == "" or overwrite_input[0].lower() == 'y':
			if conf_energy < np.nanmin(qm_energies[seen_indices]):
				print("HOORAY a new lowest energy conformer has been found.")
			qm_energies[conf_number-1] = conf_energy
		else:
			return
	if conf_number - 1 in unseen_indices:
		unseen_indices.remove(conf_number - 1)
	if len(np.where(~np.isnan(qm_energies[seen_indices]))[0]) < 5:
		init_indices = init_sampler.update_init_sample(conf_number - 1, conf_energy)
	else:
		init_indices.clear()

def probability_improvement(preds, uncert, y_min):
	z = (y_min - preds) / (uncert + 1e-9)
	return norm.cdf(z)

def fit_predict(x, y, seen_indices, unseen_indices, model):
	x_seen = x[seen_indices]
	y_seen = y[seen_indices]
	valid_indices = np.where(~np.isnan(y_seen))[0]
	x_train = x_seen[valid_indices]
	y_train = y_seen[valid_indices]
	y_mean, y_std = y_train.mean(), y_train.std()
	y_train = (y_train - y_mean) / y_std
	model.fit(x_train, y_train)
	preds, uncert = model.predict(x[unseen_indices], return_std=True)
	return preds, uncert, np.min(y_train)

def suggest_next():
	global features
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global kernel_len_scale
	print("Suggesting next conformer.")
	if not check_ready():
		print("WARNING: A minimisation has not been created or loaded. No data are available to make suggestions upon.")
		return
	if len(init_indices) > 0:
		print("The initial sample has not yet been completed. The following conformers are recommended:", end=" ")
		for index in init_indices:
			print(index + 1, end=" ")
		print()
	else:
		while True:
			batch_size = input("Enter batch size: ")
			if batch_size.lower() == "quit" or batch_size.lower() == "exit":
				return
			try:
				batch_size = int(batch_size)
			except ValueError:
				print("ERROR: Invalid batch size.")
				continue
			if batch_size <= 0:
				print("ERROR: Batch size cannot be less than zero.")
				continue
			elif batch_size > len(unseen_indices):
				print("ERROR: Batch size cannot be greater than the number of conformers remaining.")
				continue
			break
		print("Building model...")
		kernel = WhiteKernel(noise_level=0.01) + RBF(length_scale=kernel_len_scale, length_scale_bounds="fixed")
		model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, random_state=5)
		temp_y = qm_energies.copy()
		sampled_indices = list()
		temp_unseen = unseen_indices.copy()
		# Use the Kriging Believer method to select a batch of conformers
		for i in range(batch_size):
			if len(sampled_indices) >= len(unseen_indices):
				break
			y_mean = np.nanmean(temp_y[seen_indices+sampled_indices])
			y_std = np.nanstd(temp_y[seen_indices+sampled_indices])
			preds, uncert, y_min = fit_predict(features, temp_y, seen_indices + sampled_indices, temp_unseen, model)
			prob_improve = probability_improvement(preds, uncert, y_min)
			max_index = np.argmax(prob_improve)
			sampled_index = temp_unseen[max_index]
			sampled_indices.append(sampled_index)
			temp_y[sampled_index] = y_mean + y_std * preds[max_index]
			temp_unseen.pop(max_index)
		print("The following conformers are recommended: %s" % " ".join(str(sampled_index + 1) for sampled_index in sampled_indices))

def select_view_option():
	option_text = "\nSelect option (type quit/exit to return to previous menu):\n" \
	"1. Print currently calculated conformers and their energies.\n" \
	"2. Plot energy histogram.\n" \
	"3. Plot sampled energies.\n" \
	"4. Plot lowest sampled energies.\n" \
	"5. Plot skew as sampling has proceeded.\n" \
	"6. Plot exp(-(Enew - Emin)/RT) against fraction optimised.\n" \
	"7. Return to previous menu.\n"
	while True:
		selection = input(option_text)
		if selection.lower() == "quit" or selection.lower() == "exit":
			return 7
		try:
			selection = int(selection)
		except ValueError:
			print("ERROR: Input '%s' is invalid." % selection)
			continue
		if selection < 1 or selection > 7:
			print("ERROR: Input '%d' is invalid." % selection)
			continue
		else:
			break
	return selection

def view_results():
	global qm_energies
	global seen_indices
	print("Viewing results so far.")
	if not check_ready():
		print("WARNING: A minimisation has not been created or loaded. No data are available to view.")
		return
	if len(seen_indices) == 0:
		print("No energies have been collected yet.")
		return
	valid_indices = np.where(~np.isnan(qm_energies[seen_indices]))[0]
	valid_energies = qm_energies[seen_indices][valid_indices]
	while True:
		selection = select_view_option()
		if selection == 1:
			qm_min = np.min(valid_energies)
			print("Conformer | Energy (Hartree) | Rel. Energy (kcal/mol)")
			print("-----------------------------------------------------")
			for index in seen_indices:
				print("%9d | %16.7f | %16.4f" % (index + 1, qm_energies[index], 627.509 * (qm_energies[index] - qm_min)))
			print("-----------------------------------------------------")
		elif selection == 2:
			if len(valid_energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			# Use Freedman-Diaconis rule to calculate number of bins
			low_quart = np.percentile(valid_energies, 25, method="higher")
			high_quart = np.percentile(valid_energies, 75, method="lower")
			inter_range = high_quart - low_quart
			bin_width = 2 * inter_range / len(valid_energies)**(1/3)
			if bin_width == 0.0:
				print("WARNING: Not enough data for a histogram.")
				continue
			bins = np.arange(np.min(valid_energies), np.max(valid_energies) + bin_width, bin_width)
			print("Number of bins set to %d" % len(bins))
			plt.hist(valid_energies, bins=bins)
			plt.xlabel("Energy / Hartree")
			plt.title("Energy Histogram")
			plt.show()
		elif selection == 3:
			if len(valid_energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			plt.plot(np.arange(len(valid_energies)) + 1, valid_energies, "o-")
			plt.xlabel("Sample number (ignoring nan points)")
			plt.ylabel("Energy / Hartree")
			plt.title("Sampled Energies")
			plt.show()
		elif selection == 4:
			if len(valid_energies) == 0:
				print("WARNING: All energies so far are NaN.")
				continue
			min_energy = valid_energies[0]
			min_energies = list()
			for energy in valid_energies:
				if energy < min_energy:
					min_energy = energy
				min_energies.append(min_energy)
			plt.plot(np.arange(len(min_energies)) + 1, min_energies, "o-")
			plt.xlabel("Sample number (ignoring NaN points)")
			plt.ylabel("Energy / Hartree")
			plt.title("Lowest Sampled Energies")
			plt.show()
		elif selection == 5:
			if len(valid_energies) < 3:
				print("WARNING: Not enough data for skewness calculation.")
				continue
			skews = list()
			for i in range(2, len(valid_energies) + 1):
				skews.append(skew(valid_energies[:i]))
			plt.plot(np.arange(len(skews)) + 2, skews, "o-")
			plt.xlabel("Sample number (ignoring NaN points)")
			plt.ylabel("Skewness")
			plt.title("Skewness as sampling proceeded")
			plt.show()
		elif selection == 6:
			print("Make sure to cite C. C. Lam and J. M. Goodman for this idea:\nJ. Chem. Inf. Model., 2023, 63, 4364-4375, DOI: 10.1021/acs.jcim.3c00649")
			ropt = list()
			chi_new = list()
			for i, index in enumerate(seen_indices):
				ropt.append((i + 1) / qm_energies.shape[0])
				y = qm_energies[index]
				if not np.isnan(y):
					y_min = np.nanmin(qm_energies[seen_indices][:i+1])
					dg_new = 1000 * 2625.5 * (y - y_min) / (R * 298.15)
					chi_new.append(np.exp(-dg_new))
				elif len(chi_new) > 0:
					chi_new.append(chi_new[-1])
				else:
					chi_new.append(0.0)
			plt.plot(ropt, chi_new)
			plt.xlim(-0.02, 1.05)
			plt.xlabel("$r_{opt}$")
			plt.ylabel("exp(-($E_{new} - E_{min}$)/RT)")
			plt.title("exp(-($E_{new} - E_{min}$)/RT) against fraction optimised")
			plt.show()
		elif selection == 7:
			return

def save_data():
	global features
	global ff_energies
	global qm_energies
	global seen_indices
	global unseen_indices
	global init_indices
	global kernel_len_scale
	global archive_filename
	if check_ready():
		print("Saving data to archive '%s'..." % archive_filename)
		np.savez(archive_filename, features=features, ff_energies=ff_energies,
		qm_energies=qm_energies, seen_indices=seen_indices,
		unseen_indices=unseen_indices, init_indices=init_indices,
		kernel_len_scale=kernel_len_scale)
	else:
		print("WARNING: No data has been created or loaded. Not saving.")

def main():
	while True:
		selection = select_main_option()
		if selection == 1:
			start_new_job()
		elif selection == 2:
			load_prev_job()
		elif selection == 3:
			update_data()
		elif selection == 4:
			suggest_next()
		elif selection == 5:
			view_results()
		elif selection == 6:
			save_data()
		elif selection == 7:
			save_data()
			print("Exiting...")
			return

if __name__ == "__main__":
	np.random.seed(5)
	main()
