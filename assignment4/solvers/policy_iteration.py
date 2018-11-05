import time

import numpy as np
from .base import BaseSolver, one_step_lookahead


# Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
class PolicyIterationSolver(BaseSolver):
    def __init__(self, env, discount_factor=0.9, max_policy_eval_steps=None, verbose=False):
        self._env = env.unwrapped

        self._policy = np.ones([self._env.nS, self._env.nA]) / self._env.nA
        self._discount_factor = discount_factor
        self._steps = 0
        self._last_delta = 0
        self._step_times = []
        self._policy_stable = False
        self._max_policy_eval_steps = max_policy_eval_steps

        super(PolicyIterationSolver, self).__init__(verbose)

    def step(self):
        start_time = time.clock()
        # Evaluate the current policy
        V = self.evaluate_policy(self._policy, discount_factor=self._discount_factor,
                                 max_steps=self._max_policy_eval_steps)

        # Will be set to false if we make any changes to the policy
        self._policy_stable = True

        delta = 0
        reward = 0  # float('-inf')
        # For each state...
        for s in range(self._env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(self._policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitrarily
            action_values = one_step_lookahead(self._env, self._discount_factor, s, V)
            best_a = np.argmax(action_values)
            best_action_value = np.max(action_values)

            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # reward = max(reward, best_action_value)
            reward += best_action_value

            # Greedily update the policy
            if chosen_a != best_a:
                self._policy_stable = False
            self._policy[s] = np.eye(self._env.nA)[best_a]

        # If we've gone through a few steps but have not improved, consider us converged
        if delta > self._last_delta and self._steps > 10:
            self._policy_stable = True

        self._steps += 1
        self._step_times.append(time.clock() - start_time)
        self._last_delta = delta

        return self._policy, V, self._steps, self._step_times[-1], reward, delta, self._policy_stable

    def reset(self):
        self._policy = np.ones([self._env.nS, self._env.nA]) / self._env.nA
        self._steps = 0
        self._step_times = []
        self._last_delta = 0
        self._policy_stable = False

    def has_converged(self):
        return self._policy_stable

    def get_convergence(self):
        return self._last_delta

    def run_until_converged(self):
        while not self.has_converged():
            self.step()

    def get_environment(self):
        return self._env
