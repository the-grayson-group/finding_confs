import os
import sys
import numpy as np

if len(sys.argv) != 2:
	print("ERROR Usage: analyse_results.py <results_file>")
	exit(1)
if not os.path.exists(sys.argv[1]):
	print(f"ERROR: '{sys.argv[1]}' - no such file or directory.")
	exit(1)
elif not os.path.isfile(sys.argv[1]):
	print(f"ERROR: '{sys.argv[1]}' is not a file.")
	exit(1)
elif not os.access(sys.argv[1], os.R_OK):
	print(f"ERROR: Cannot open '{sys.argv[1]}' - permission denied.")
	exit(1)
acq_func = (0, 1, 2, 3)
init_scheme = (0, 1, 2)
ff_treat = (0, 1, 2)
n_dims = (3, 4, 6, 10, 20, 30)
acq_names = ("Exploit", "LCB", "PI", "EI")
init_names = ("FFLow", "FFSpread", "Cluster")
ff_treat_names = ("Ignore", "CompressWith", "AfterCompress")
mol_results = dict()
lowest_sums = dict()
analysed = False
with open(sys.argv[1], "r") as file:
	for line in file:
		line = line.strip().split()
		if line[0] == "#":
			if line[1].isnumeric():
				indices = tuple([int(val) for val in line[1:4]] + [n_dims.index(int(line[4]))])
				expt_string = " ".join([acq_names[indices[0]], init_names[indices[1]], ff_treat_names[indices[2]], str(n_dims[indices[3]])])
				if expt_string not in mol_results[mol_name]:
					mol_results[mol_name][expt_string] = 0
				if expt_string not in lowest_sums[mol_name]:
					lowest_sums[mol_name][expt_string] = 0.0
				analysed = False
			else:
				mol_name = os.path.basename(line[1])
				if mol_name not in mol_results:
					mol_results[mol_name] = dict()
				if mol_name not in lowest_sums:
					lowest_sums[mol_name] = dict()
		else:
			if float(line[1]) == 0.0 and not analysed:
				mol_results[mol_name][expt_string] += int(line[0])
				analysed = True
			if int(line[0]) >= 5:
				lowest_sums[mol_name][expt_string] += float(line[1])

results = dict()
for mol_name in mol_results:
	for expt_string in mol_results[mol_name]:
		if expt_string not in results:
			results[expt_string] = mol_results[mol_name][expt_string]
		else:
			results[expt_string] += mol_results[mol_name][expt_string]
		print(mol_name, expt_string, mol_results[mol_name][expt_string])

best_expt = ""
lowest_count = 100000
for expt_string in results:
	if results[expt_string] < lowest_count:
		lowest_count = results[expt_string]
		best_expt = expt_string
	print(expt_string, results[expt_string])
print("Best method:", best_expt, "samples =", lowest_count)

best_low_total = 0.0
for mol_name in lowest_sums:
	best_low_total += lowest_sums[mol_name][best_expt] / 25
	print(mol_name, best_expt, "%f" % (lowest_sums[mol_name][best_expt] / 25))
print("Total lowest sum for", best_expt, "= %.2f" % best_low_total)
