import os
import sys
import random
import pickle
import numpy as np
from file_utils import get_conformers_filenames
from benchmark_forcefield import pipeline_mix_opt_data_all
from stop_predictor import train_stop_predictor


if __name__ == "__main__":
	random.seed(5)
	if len(sys.argv) < 2:
		print("ERROR Usage: python cross_validate_methods.py [sdf_files]")
		exit(1)
	ff_sdf_files, dft_sdf_files, ff_energy_files, dft_energy_files = \
		get_conformers_filenames()
	if len(ff_sdf_files) == 0:
		print("ERROR No valid files were provided.")
		exit(1)
	opt_features_all, stop_labels_all, min_energies_all = \
		pipeline_mix_opt_data_all(ff_sdf_files, dft_sdf_files, ff_energy_files,
		dft_energy_files)
	opt_features = list()
	stop_labels = list()
	for features, labels in zip(opt_features_all, stop_labels_all):
		opt_features.extend(features)
		stop_labels.extend(labels)
	opt_features = np.array(opt_features)
	stop_labels = np.array(stop_labels)
	stop_predictor = train_stop_predictor(opt_features, stop_labels)
	data_name = os.path.basename(os.path.dirname(ff_sdf_files[0]))
	stop_predictor_file = open(data_name + "_stop_pred.pkl", "wb")
	pickle.dump(stop_predictor, stop_predictor_file)
