import os
import sys
import numpy as np

def minimise_conf_energy(feature_file, dft_file, ff_file):
	batch_size = 1
	print(f"# {feature_file}")
	print("# 0 0 0 0")
	molecule_name = feature_file.replace("_data.npy", "")
	x = np.load(ff_file)
	y = np.load(dft_file)
	target_y = np.min(y[np.where(~np.isnan(y))])
	min_y = 1000000.0
	last_y = min_y
	sorted_indices = np.argsort(x)
	i = 0
	while i < len(sorted_indices):
		sampled_indices = list()
		j = 0
		while i < len(sorted_indices) and j < batch_size:
			sampled_indices.append(sorted_indices[i])
			i += 1
			j += 1
		for index in sampled_indices:
			if not np.isnan(y[index]):
				last_y = 627.509 * (y[index] - target_y)
				if last_y < min_y:
					min_y = last_y
		print(f"{i} {min_y:.5f} {last_y:.5f}")

def get_data_cmdline():
	feature_data_files = list()
	dft_data_files = list()
	ff_data_files = list()
	for filename in sys.argv[1:]:
		if os.path.isdir(filename):
			for data_filename in os.listdir(filename):
				if data_filename.endswith("_data.npy"):
					data_filename = os.path.join(filename, data_filename)
					dft_filename = data_filename.replace("_data.npy", "_dft.npy")
					ff_filename = data_filename.replace("_data.npy", "_ff.npy")
					if os.path.exists(dft_filename) and os.path.exists(ff_filename):
						feature_data_files.append(data_filename)
						dft_data_files.append(dft_filename)
						ff_data_files.append(ff_filename)
					else:
						continue
		elif filename.endswith("_data.npy"):
			dft_filename = filename.replace("_data.npy", "_dft.npy")
			ff_filename = filename.replace("_data.npy", "_ff.npy")
			if os.path.exists(dft_filename) and os.path.exists(ff_filename):
				feature_data_files.append(filename)
				dft_data_files.append(dft_filename)
				ff_data_files.append(ff_filename)
	return feature_data_files, dft_data_files, ff_data_files

if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("ERROR: Need to provide data files.")
		exit(1)
	feature_files, dft_files, ff_files = get_data_cmdline()
	for i in range(len(feature_files)):
		minimise_conf_energy(feature_files[i], dft_files[i], ff_files[i])
