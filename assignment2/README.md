
# Assignment 2 - Randomized Optimization

The code for this assignment chooses three toy problems, but there are other options available in _ABAGAIL_. 

## General

If you are running this code in OS X you should consider downloading Jython directly. The version provided by homebrew does mot seem to work as expected for this code.

On windows, install Jython 2.7.0 as the 2.7.1 can have issues. Used the installer version with Standard install to C:\jython2.7.0 path for simplicity.

## Data

The data loading code expects datasets to be stored in "./data".

Because _ABAGAIL_ does not implement cross validation some work must be done on the dataset before the other code can
be run. The data can be generated via 

```
python run_experiment.py --dump_data
```
 
Be sure to run this before running any of the experiments.

## Output

Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders will be created for
each toy problem (`CONTPEAKS`, `FLIPFLOP`, `TSP`) and the neural network from the _Supervised Learning Project_ (`NN_OUTPUT`, `NN`).

If these folders do not exist the experiments module will attempt to create them.

## Running Experiments

Each experiment can be run as a separate script. Running the actual optimization algorithms to generate data requires
the use of Jython.

For the three toy problems, run:
 - continuoutpeaks.py
 - flipflop.py
 - tsp.py

For the neural network problem, run:
 - NN-Backprop.py
 - NN-GA.py
 - NN-RHC.py
 - NN-SA.py

## Graphing

The `plotting.py` script takes care of all the plotting. Since the files output from the scripts above follow a common
naming scheme it will determine the problem, algorithm, and parameters as needed and write the output to sub-folders in
`./output/images`. This _must_ be run via python, specifically an install of python that has the requirements from
`requirements.txt` installed.

In addition to the images, a csv file of the best parameters per problem/algorithm pair is written to
`./output/best_results.csv`
