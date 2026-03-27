The scripts in this folder perform the main experiments on our algorithms for conformational energy minimization. In order to use them, they require the data files from the [University of Bath Research Data Archive](https://researchdata.bath.ac.uk/1492). For each of the folders `molecules`, `crest_molecules`, `transition_states` and `single_points`, download the corresponding zip archive file with the same name and extract its contents. Additionally, the C code within `geometry.c` needs to be compiled into a shared object library `libgeometry.so`, which can be done by running the `make` command in this folder, or through the command `cc -fPIC -shared -o libgeometry.so geometry.c -lm`.


## Baseline Experiments
The scripts that are relevant to the experiments performed using the baseline comparison method based on Pipeline-Mix from [CONFPASS](https://doi.org/10.1021/acs.jcim.3c00649) are `benchmark_forcefield.py` and `test_forcefield.py`.

To perform the experiments which test different thresholds for the stop predictor in the Pipeline-Mix method on the force field-searched "tuning" molecules and then the CREST-searched molecules, run the following:
```
python benchmark_forcefield.py PATH_TO_FOLDER/molecules/
python benchmark_forcefield.py PATH_TO_FOLDER/crest_molecules/
```

To perform the experiments in which the Pipeline-Mix method is tested on the "unseen" transition state data sets, the stop predictor model must first be trained on all of the conformer sets in the molecules folder by running:
```
python train_full_stop_pred.py PATH_TO_FOLDER/molecules/
```

Then the experiments can be performed on the transition state conformer sets (using the force field energies, then single-point DFT energies as the low level of theory):
```
python test_forcefield.py PATH_TO_FOLDER/transition_states/
python test_forcefield.py PATH_TO_FOLDER/single_points/
```

The stop predictor probability threshold for these experiments can be changed with the `confidence` argument in the final line of `test_forcefield.py`.


## Bayesian Optimization Experiments
The scripts that are relevant to the Bayesian optimization experiments are `benchmark_bayesian.py`, `test_bayesian.py`, `benchmark_batching.py`, `test_batching.py` and `test_selectivity.py`.

To perform the experiments in which the Bayesian optimization settings (features, initial sample method and inclusion/exclusion of low level energy feature) are tested on the data set of force field-searched "tuning" molecules and CREST-searched molecules, run the following:
```
python benchmark_bayesian.py PATH_TO_FOLDER/molecules/
python benchmark_bayesian.py PATH_TO_FOLDER/crest_molecules/
```

To perform the experiments in which Bayesian optimization is tested on the "unseen" transition state data sets, using the force field energies, then single-point DFT energies as the low level of theory, run the following:
```
python test_bayesian.py PATH_TO_FOLDER/transition_states/
python test_bayesian.py PATH_TO_FOLDER/single_points/
```

To perform the experiments in which different batch sizes are used with Bayesian optimization on the tuning molecules, run the following:
```
python benchmark_batching.py PATH_TO_FOLDER/molecules/
```

To perform the experiments in which batched Bayesian optimization is applied to the transition state data sets (the batch size may be changed by adjusting the `BATCH_SIZE` global variable), run the following:
```
python test_batching.py PATH_TO_FOLDER/transition_states/
```

Finally, to perform the experiments in which Bayesian optimization is used to find the lowest energy conformers of both of the diastereomers of the transition states, run the following:
```
python test_selectivity.py PATH_TO_FOLDER/transition_states/
```
