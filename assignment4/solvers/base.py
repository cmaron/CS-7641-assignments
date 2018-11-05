import csv
import logging

import numpy as np

from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpisodeStats(object):
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_lengths = np.zeros(num_episodes)
        self.episode_times = np.zeros(num_episodes)
        self.episode_rewards = np.zeros(num_episodes)
        self.episode_deltas = np.zeros(num_episodes)

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            f.write("episode,length,time,reward,delta\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(range(self.num_episodes), self.episode_lengths, self.episode_times,
                                 self.episode_rewards, self.episode_deltas))

    @staticmethod
    def from_df(df):
        es = EpisodeStats(df.shape[0])
        es.episode_lengths = df['length'].values
        es.episode_times = df['time'].values
        es.episode_rewards = df['reward'].values
        es.episode_deltas = df['delta'].values


def one_step_lookahead(env, discount_factor, state, v):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        v: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * v[next_state])
    return A


class BaseSolver(ABC):
    def __init__(self, verbose=False):
        self._verbose = verbose

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def has_converged(self):
        pass

    @abstractmethod
    def get_convergence(self):
        pass

    @abstractmethod
    def run_until_converged(self):
        pass

    @abstractmethod
    def get_environment(self):
        pass

    # TODO: Move this?
    # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    def evaluate_policy(self, policy, discount_factor=1.0, max_steps=None, theta=0.00001):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.

        Args:
            policy: The policy to evaluate
            max_steps: If not none, the number of iterations to run
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            Vector of length env.nS representing the value function.
        """
        env = self.get_environment()
        # Start with a random (all 0) value function
        V = np.zeros(env.nS)
        steps = 0
        while max_steps is None or steps < max_steps:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(env.nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]):
                    # For each action, look at the possible next states...
                    for prob, next_state, reward, done in env.P[s][a]:
                        # Calculate the expected value
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            # print('{} {} {}'.format(steps, delta, v))
            steps += 1
            # print("delta: {}, theta: {}".format(delta, theta))
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break

        return np.array(V)

    # TODO: Move this elsewhere?
    def render_policy(self, policy):
        env = self.get_environment()
        directions = env.directions()
        policy = np.reshape(np.argmax(policy, axis=1), env.desc.shape)

        for row in range(policy.shape[0]):
            for col in range(policy.shape[1]):
                print(directions[policy[row, col]] + ' ', end="")
            print("")

    # TODO: Move this elsewhere?
    def run_policy(self, policy, max_steps=1000, render_during=False):
        """
        Run through the given policy. This will reset the solver's environment before running.

        :param policy: The policy to run
        :param max_steps: The total number of steps to run. This helps prevent the agent getting "stuck"
        :param render_during: If true, render the env to stdout at each step
        :return: An ndarray of rewards for each step
        """
        policy = np.argmax(policy, axis=1)

        rewards = []

        # Clone the environment to get a fresh one
        env = self.get_environment().new_instance()
        state = env.reset()

        done = False
        steps = 0
        while not done and steps < max_steps:
            if render_during:
                env.render()

            action = policy[state]
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1

        if render_during:
            env.render()

        return np.array(rewards)

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))

