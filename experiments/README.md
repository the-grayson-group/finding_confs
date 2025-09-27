The `*finding_confs.py` scripts in this folder perform the main experiments on our algorithms for conformational energy minimization. In order to use them, these scripts will require the data files from the [University of Bath Research Data Archive](https://researchdata.bath.ac.uk/1492). For each of the folders `goodman_molecules`, `grayson_molecules` and `ts_data`, download the corresponding zip archive file with the same name and extract its contents into that folder. Additionally, the C code within `geometry.c` needs to be compiled into a shared object library `libgeometry.so`, which can be done by running the `make` command in this folder, or through the command `cc -fPIC -shared -o libgeometry.so geometry.c -lm`.

To perform the experiment which tests the performances of the various design parameters of the Bayesian optimization algorithm on the molecule datasets, run `finding_confs.py` as follows:
```
python finding_confs.py goodman_molecules/ grayson_molecules/
```
This will write the output logs to a file name `goodman_molecules_find_conf_results.txt` (which may also be found in the [University of Bath Research Data Archive](https://researchdata.bath.ac.uk/1492)). Note that this program can take a while to run since it tests every combination of design parameter settings on all of the molecules, and also repeats 25 times.

To perform the experiment which tests the different batched acquisition functions in combination with the best design parameters optimized above using the molecule datasets, run `batch_finding_confs.py` as follows:
```
python batch_finding_confs.py goodman_molecules/ grayson_molecules/
```

To perform the experiment using only the force field to select conformers, run `ff_finding_confs.py` as follows:
```
python ff_finding_confs.py goodman_molecules/ grayson_molecules/
```
The batch size used by the force field sampling may be changed by editing line 6 of `ff_finding_confs.py`.

To perform the validation of the tuned algorithm on the transition state data sets, run `print_finding_confs.py` as follows:
```
python print_finding_confs.py ts_data/
```
Note that the `*finding_confs.py` scripts can also be provided with the `*_data.npy` files for individual systems, so long as the `*_ff.npy` and `*_dft.npy` files are also present in the same directory. Thus, one could also check the results of the best algorithm for just one of the chemical systems, for example:
```
python print_finding_confs.py ts_data/BPA_TS_data.npy
```

The format of the output of all of these scripts is: number of conformer samples taken so far, lowest energy sampled so far (relative to global minimum, kcal/mol), relative energy of latest sampled conformer (kcal/mol).
