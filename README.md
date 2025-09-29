# finding_confs
Source code for the paper "Alleviating the Conformational Bottleneck in Mechanistic Modeling with Bayesian Optimization".

## Software Environment
This code was developed and tested using Python 3.10.12 and GCC 11.4.0 on x86_64 Ubuntu Linux 22.04 running on a workstation with an Intel Xeon Gold 6238R CPU. The specific versions of the Python libraries used (importantly, scikit-learn, numpy, scipy, rdkit) may be found in `requirements.txt`.

## Experiments
The `experiments` folder contains the source code for the programs that perform our experiments using Bayesian optimization to minimize the conformational energies of chemical systems. Details of how these are used may be found in `experiments/README.md`.

## Analysis
The `analysis` folder contains the scripts that were used to analyse the outputs of the programs in the `experiments` folder to produce summaries of the algorithm performances. Information on how these scripts are used may be found in `analysis/README.md`.

## minconf.py
This program is a stand-alone implementation of the best-performing variation of our Bayesian optimization algorithm as a simple user interface, allowing the use of our algorithm to find the lowest-energy conformers of chemical systems "in real time", after one has performed an initial conformational search. Here, we detail its usage.

### Setup
Firstly, the C library that calculates the reciprocal distance matrix representation of the conformers and the bandwidth parameter for the Gaussian process regression kernel must be compiled. If GCC and GNU Make are available, run the `make` command with `Makefile`, `geometry.c` and `minconf.py` all in the same folder. If all goes well the `libgeometry.so` library will be present and `minconf.py` will be ready to use. Otherwise, use your favourite C compiler, and run the command:
```
cc -fPIC -shared -o libgeometry.so geometry.c -lm
```
We have successfully compiled this C code on Linux and macOS, however we have not tested it on Windows. Whilst it is hopefully simple enough to be relatively portable, we cannot make any guarantees about whether it will compile and run without issue on Windows.

`minconf.py` requires all of the Python libraries listed in `requirements.txt`. No extremely low-level API calls are made to these libraries, thus this program's stability should hopefully be relatively insensitive to slightly newer version numbers.

### Usage
Upon running `minconf.py` from the command line, the user will be presented with a numerical menu providing options for what to do. Entering option `1` will begin a new conformer energy optimization "job". For this, the user will require an SDF file containing all of the conformers of the chemical system of interest and a pre-prepared NumPy file containing a single, one-dimensional array of the low-level energies used in the initial conformational search, *in the order of the conformers in the SDF file*. The program will then compute all the values it needs (which may take several moments for large systems or high numbers of conformers) and it will save its data to a file named after the base name of the SDF file plus the extension ".cnfmin.npz". Option 2 will allow the user to load their created .cnfmin.npz file and continue their work. Option 3 is used to input a calculated conformer energy to be used as training data for the surrogate machine learning model in the Bayesian optimization routine. This is done by first entering the number of the conformer for which one has an energy value (note that the conformers are numbered starting from 1) followed by its energy. Additionally, the user can input "nan" if a particular conformer cannot be computed and is to be removed from the search (e.g. calculations cannot be converged), or "del" if a conformer is to be added back to the set from which the Bayesian optimization may select. Option 4 is used to query the program for its recommendations for which conformer(s) is to be modeled next. Before the initial sample of size 5 is acquired, the program will suggest conformers based on those with the lowest low level energies. After the initial sample is complete, the user will be prompted for a batch size of the number of conformers they wish to be recommended together and the program will provide the conformer numbers with the highest ranked probability of improvement scores. Option 5 allows the user to view the results they have collected so far, including a table of the raw input data and summary plots of the collected energies as the sampling has proceeded. Options 6 and 7 respectively save the data to the .cnfmin.npz file that was loaded or created earlier, and save and exit the program. Entering "exit" or "quit" anywhere in the program will move back to the previous menu (or save and exit the program if at the top level) before any data are written during usage of a submenu.
