import csv
import logging
import os
import math
import pickle
import time

import numpy as np

from abc import ABC, abstractmethod

from .plotting import plot_policy_map, plot_value_map
import solvers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

MAX_STEP_COUNT = 1000


class EvaluationStats(object):
    def __init__(self):
        self.rewards = list()
        self.stat_history = list()
        self.reward_mean = 0
        self.reward_median = 0
        self.reward_std = 0
        self.reward_max = 0
        self.reward_min = 0
        self.runs = 0

    def add(self, reward):
        self.rewards.append(reward)
        self.compute()

    def compute(self):
        reward_array = np.array(self.rewards)
        self.runs = len(self.rewards)
        self.reward_mean = np.mean(reward_array)
        self.reward_median = np.median(reward_array)
        self.reward_std = np.std(reward_array)
        self.reward_max = np.max(reward_array)
        self.reward_min = np.min(reward_array)
        self.stat_history.append((
            self.reward_mean,
            self.reward_median,
            self.reward_std,
            self.reward_max,
            self.reward_min
        ))

    def to_csv(self, file_name):
        self.compute()
        means, medians, stds, maxes, mins = zip(*self.stat_history)
        with open(file_name, 'w') as f:
            f.write("step,reward,mean,median,std,max,min\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(range(len(self.rewards)), self.rewards, means, medians, stds, maxes, mins))

    def __str__(self):
        return 'reward_mean: {}, reward_median: {}, reward_std: {}, reward_max: {}, reward_min: {}, runs: {}'.format(
            self.reward_mean,
            self.reward_median,
            self.reward_std,
            self.reward_max,
            self.reward_min,
            self.runs
        )


class ExperimentStats(object):
    def __init__(self):
        self.policies = list()
        self.vs = list()
        self.steps = list()
        self.step_times = list()
        self.rewards = list()
        self.deltas = list()
        self.converged_values = list()
        self.elapsed_time = 0
        self.optimal_policy = None

    def add(self, policy, v, step, step_time, reward, delta, converged):
        self.policies.append(policy)
        self.vs.append(v)
        self.steps.append(step)
        self.step_times.append(step_time)
        self.rewards.append(reward)
        self.deltas.append(delta)
        self.converged_values.append(converged)

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            f.write("steps,time,reward,delta,converged\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.steps, self.step_times, self.rewards, self.deltas, self.converged_values))

    def pickle_results(self, file_name_base, map_shape, step_size=1, only_last=False):
        if only_last:
            policy = np.reshape(np.argmax(self.policies[-1], axis=1), map_shape)
            v = self.vs[-1].reshape(map_shape)
            file_name = file_name_base.format('Last')
            with open(file_name, 'wb') as f:
                pickle.dump({'policy': policy, 'v': v}, f)
        else:
            l = len(self.policies)
            if step_size == 1 and l > 20:
                step_size = math.floor(l/20.0)
            for i, policy in enumerate(self.policies):
                if i % step_size == 0 or i == l-1:
                    v = self.vs[i].reshape(map_shape)
                    file_name = file_name_base.format(i)
                    if i == l-1:
                        file_name = file_name_base.format('Last')
                    with open(file_name, 'wb') as f:
                        pickle.dump({'policy': np.reshape(np.argmax(policy, axis=1), map_shape), 'v': v}, f)

    def plot_policies_on_map(self, file_name_base, map_desc, color_map, direction_map, experiment, step_preamble,
                             details, step_size=1, only_last=False):
        if only_last:
            policy = np.reshape(np.argmax(self.policies[-1], axis=1), map_desc.shape)
            v = self.vs[-1].reshape(map_desc.shape)

            policy_file_name = file_name_base.format('Policy', 'Last')
            value_file_name = file_name_base.format('Value', 'Last')
            title = '{}: {} - {} {}'.format(details.env_readable_name, experiment, 'Last', step_preamble)

            p = plot_policy_map(title, policy, map_desc, color_map, direction_map)
            p.savefig(policy_file_name, format='png', dpi=150)
            p.close()

            p = plot_value_map(title, v, map_desc, color_map)
            p.savefig(value_file_name, format='png', dpi=150)
            p.close()
        else:
            l = len(self.policies)
            if step_size == 1 and l > 20:
                step_size = math.floor(l/20.0)
            for i, policy in enumerate(self.policies):
                if i % step_size == 0 or i == l-1:
                    policy = np.reshape(np.argmax(policy, axis=1), map_desc.shape)
                    v = self.vs[i].reshape(map_desc.shape)

                    file_name = file_name_base.format('Policy', i)
                    value_file_name = file_name_base.format('Value', i)
                    if i == l-1:
                        file_name = file_name_base.format('Policy', 'Last')
                        value_file_name = file_name_base.format('Value', 'Last')

                    title = '{}: {} - {} {}'.format(details.env_readable_name, experiment, step_preamble, i)

                    p = plot_policy_map(title, policy, map_desc, color_map, direction_map)
                    p.savefig(file_name, format='png', dpi=150)
                    p.close()

                    p = plot_value_map(title, v, map_desc, color_map)
                    p.savefig(value_file_name, format='png', dpi=150)
                    p.close()

    def __str__(self):
        return 'policies: {}, vs: {}, steps: {}, step_times: {}, deltas: {}, converged_values: {}'.format(
            self.policies,
            self.vs,
            self.steps,
            self.step_times,
            self.deltas,
            self.converged_values
        )


class ExperimentDetails(object):
    def __init__(self, env, env_name, env_readable_name, threads, seed):
        self.env = env
        self.env_name = env_name
        self.env_readable_name = env_readable_name
        self.threads = threads
        self.seed = seed


class BaseExperiment(ABC):
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    @abstractmethod
    def perform(self):
        pass

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))

    def run_solver_and_collect(self, solver, convergence_check_fn):
        stats = ExperimentStats()

        t = time.clock()
        step_count = 0
        optimal_policy = None
        best_reward = float('-inf')

        while not convergence_check_fn(solver, step_count) and step_count < MAX_STEP_COUNT:
            policy, v, steps, step_time, reward, delta, converged = solver.step()
            # print('{} {}'.format(reward, best_reward))
            if reward > best_reward:
                best_reward = reward
                optimal_policy = policy

            stats.add(policy, v, steps, step_time, reward, delta, converged)
            # if self._verbose:
            #     self.log("Step {}: delta={}, converged={}".format(step_count, delta, converged))
            step_count += 1

        stats.elapsed_time = time.clock() - t
        stats.optimal_policy = stats.policies[-1]  # optimal_policy
        return stats

    def run_policy_and_collect(self, solver, policy, times=100):
        stats = EvaluationStats()
        for i in range(times):
            # stats.add(np.sum(solver.run_policy(policy)))
            stats.add(np.mean(solver.run_policy(policy)))
        stats.compute()

        return stats