## Analysis
The scripts in this folder were used to calculate the summaries of the performances of the various conformational energy minimization algorithms we test from the outputs of the programs in the `experiments` folder. These scripts require the output log files that are available from the [University of Bath Research Data Archive](https://researchdata.bath.ac.uk/1492), in the archive file `log_files.zip`. Each script should be used with a specific log file, as follows:
```
python analyse_results.py goodman_molecules_find_conf_results.txt
python analyse_batch_results.py batch_goodman_molecules_results.txt
python analyse_ff_results.py ff_find_conf_results.txt
python analyse_ff_results.py batch_ff_find_conf_results.txt
``` 
