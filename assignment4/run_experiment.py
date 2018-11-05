import argparse
from datetime import datetime
import logging

import random as rand
import numpy as np

import environments
import experiments

from experiments import plotting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the policy iteration experiment')
    parser.add_argument('--value', action='store_true', help='Run the value iteration experiment')
    parser.add_argument('--q', action='store_true', help='Run the Q-Learner experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
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

    logger.info("Creating MDPs")
    logger.info("----------")

    envs = [
        {
            # This is not really a rewarding frozen lake env, but the custom class has extra functionality
            'env': environments.get_rewarding_no_reward_frozen_lake_environment(),
            'name': 'frozen_lake',
            'readable_name': 'Frozen Lake (8x8)',
        },
        {
            'env': environments.get_large_rewarding_no_reward_frozen_lake_environment(),
            'name': 'large_frozen_lake',
            'readable_name': 'Frozen Lake (20x20)',
        },
        {
            'env': environments.get_windy_cliff_walking_environment(),
            'name': 'cliff_walking',
            'readable_name': 'Cliff Walking (4x12)',
        }
    ]

    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            threads=threads,
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    logger.info("Running experiments")

    timings = {}

    if args.policy or args.all:
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings)

    if args.value or args.all:
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings)

    if args.q or args.all:
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'Q', verbose, timings)

    logger.info(timings)

    if args.plot:
        if verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results(envs)
