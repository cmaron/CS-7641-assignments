import argparse
from datetime import datetime
import logging

import random as rand
import numpy as np

from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.ds_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for Randomized Optimization')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--dump_data', action='store_true', help='Build train/test/validate splits '
                                                                 'and save to the data folder '
                                                                 '(should only need to be done once)')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Loading data")
    logger.info("----------")

    datasets = [
        # {
        #     'data': loader.StatlogVehicleData(verbose=verbose, seed=seed),
        #     'name': 'statlog_vehicle',
        #     'readable_name': 'Statlog Vehicle',
        # },
        # {
        #     'data': loader.HTRU2Data(verbose=verbose, seed=seed),
        #     'name': 'htru2',
        #     'readable_name': 'HTRU2',
        # },
        {
            'data': loader.CreditApprovalData(verbose=verbose, seed=seed),
            'name': 'credit_approval',
            'readable_name': 'Credit Approval',
        },
        {
            'data': loader.PenDigitData(verbose=verbose, seed=seed),
            'name': 'pen_digits',
            'readable_name': 'Handwritten Digits',
        }
        # {
        #     'data': loader.SpamData(verbose=verbose, seed=seed),
        #     'name': 'spam',
        #     'readable_name': 'Spam',
        # },
        # {
        #     'data': loader.CreditDefaultData(verbose=verbose, seed=seed),
        #     'name': 'credit_default',
        #     'readable_name': 'Credit Default',
        # }
    ]

    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        if args.dump_data:
            data.dump_test_train_val(test_size=0.2, random_state=seed)

