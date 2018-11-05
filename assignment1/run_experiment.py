import argparse
from datetime import datetime
import numpy as np

import experiments
from data import loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some SL experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        print("Using seed {}".format(seed))

    print("Loading data")
    print("----------")

    ds1_data = loader.CreditDefaultData(verbose=verbose, seed=seed)
    ds1_name = 'credit_default'
    ds1_readable_name = 'Credit Default'
    ds1_data.load_and_process()

    ds2_data = loader.PenDigitData(verbose=verbose, seed=seed)
    ds2_name = 'pen_digits'
    ds2_readable_name = 'Handwritten Digits'
    ds2_data.load_and_process()

    if verbose:
        print("----------")
    print("Running experiments")

    timings = {}

    experiment_details_ds1 = experiments.ExperimentDetails(
        ds1_data, ds1_name, ds1_readable_name,
        threads=threads,
        seed=seed
    )

    experiment_details_ds2 = experiments.ExperimentDetails(
        ds2_data, ds2_name, ds2_readable_name,
        threads=threads,
        seed=seed
    )

    if args.ann or args.all:
        t = datetime.now()
        experiment = experiments.ANNExperiment(experiment_details_ds1, verbose=verbose)
        experiment.perform()
        experiment = experiments.ANNExperiment(experiment_details_ds2, verbose=verbose)
        experiment.perform()
        t_d = datetime.now() - t
        timings['ANN'] = t_d.seconds

    if args.boosting or args.all:
        t = datetime.now()
        experiment = experiments.BoostingExperiment(experiment_details_ds1, verbose=verbose)
        experiment.perform()
        experiment = experiments.BoostingExperiment(experiment_details_ds2, verbose=verbose)
        experiment.perform()
        t_d = datetime.now() - t
        timings['Boost'] = t_d.seconds

    if args.dt or args.all:
        t = datetime.now()
        experiment = experiments.DTExperiment(experiment_details_ds1, verbose=verbose)
        experiment.perform()
        experiment = experiments.DTExperiment(experiment_details_ds2, verbose=verbose)
        experiment.perform()
        t_d = datetime.now() - t
        timings['DT'] = t_d.seconds

    if args.knn or args.all:
        t = datetime.now()
        experiment = experiments.KNNExperiment(experiment_details_ds1, verbose=verbose)
        experiment.perform()
        experiment = experiments.KNNExperiment(experiment_details_ds2, verbose=verbose)
        experiment.perform()
        t_d = datetime.now() - t
        timings['KNN'] = t_d.seconds

    if args.svm or args.all:
        t = datetime.now()
        experiment = experiments.SVMExperiment(experiment_details_ds1, verbose=verbose)
        experiment.perform()
        experiment = experiments.SVMExperiment(experiment_details_ds2, verbose=verbose)
        experiment.perform()
        t_d = datetime.now() - t
        timings['SVM'] = t_d.seconds

    print(timings)
