import time
import numpy as np

from .base import BaseSolver, one_step_lookahead


# Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
class ValueIterationSolver(BaseSolver):
    # Originally 0.0001, not 0.00001
    def __init__(self, env, discount_factor=0.9, theta=0.00001, verbose=False):
        self._env = env.unwrapped

        self._V = np.zeros(self._env.nS)

        self._policy = np.zeros([self._env.nS, self._env.nA])
        self._discount_factor = discount_factor
        self._theta = theta
        self._steps = 0
        self._last_delta = theta
        self._step_times = []

        super(ValueIterationSolver, self).__init__(verbose)

    def step(self):
        start_time = time.clock()

        delta = 0
        reward = 0
        # Update each state...
        for s in range(self._env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(self._env, self._discount_factor, s, self._V)

            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - self._V[s]))
            # reward = max(reward, best_action_value)
            reward += best_action_value

            # Update the value function. Ref: Sutton book eq. 4.10.
            self._V[s] = best_action_value
        self._step_times.append(time.clock() - start_time)

        self._last_delta = delta
        self._steps += 1

        # Create a deterministic policy using the optimal value function
        self._policy = np.zeros([self._env.nS, self._env.nA])
        for s in range(self._env.nS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(self._env, self._discount_factor, s, self._V)
            best_action = np.argmax(A)
            # Always take the best action
            self._policy[s, best_action] = 1.0

        return self._policy, self._V, self._steps, self._step_times[-1], reward, delta, self.has_converged()

    def reset(self):
        self._V = np.zeros(self._env.nS)
        self._policy = np.zeros([self._env.nS, self._env.nA])
        self._steps = 0
        self._step_times = []
        self._last_delta = 0

    def has_converged(self):
        return self._last_delta < self._theta

    def get_convergence(self):
        return self._last_delta

    def run_until_converged(self):
        while not self.has_converged():
            self.step()

    def get_environment(self):
        return self._env
