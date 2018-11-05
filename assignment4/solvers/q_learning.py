import time
import numpy as np

from .base import BaseSolver, one_step_lookahead, EpisodeStats


# Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
class QLearningSolver(BaseSolver):
    def __init__(self, env, max_episodes, max_steps_per_episode=500, discount_factor=1.0, alpha=0.5, epsilon=0.1,
                 epsilon_decay=0.001, q_init=0, theta=0.0001, min_consecutive_sub_theta_episodes=10, verbose=False):
        self._env = env.unwrapped

        self._max_episodes = max_episodes
        # Require we run for at least 5% of the max number of episodes
        self._min_episodes = np.floor(max_episodes * 0.1)
        self._max_steps_per_episode = max_steps_per_episode
        self._epsilon = epsilon
        self._initial_epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._alpha = alpha
        self._discount_factor = discount_factor
        self._q_init = q_init
        self._steps = 0
        self._step_times = []
        self._last_delta = 0
        self._theta = theta
        self._stats = EpisodeStats(max_episodes)

        # We want to wait for a few consecutive episodes to be below theta before we consider the model converged
        self._consecutive_sub_theta_episodes = 0
        self._min_consecutive_sub_theta_episodes = min_consecutive_sub_theta_episodes

        self._init_q()

        super(QLearningSolver, self).__init__(verbose)

    def step(self):
        start_time = time.clock()

        # Reset the environment and pick the first action
        state = self._env.reset()

        # One step in the environment
        total_reward = 0.0
        episode_steps = 0
        for t in range(self._max_steps_per_episode+1):
            # Take a step
            action_probs = self._policy_function(state)
            # TODO: Which one?
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self._env.step(action)

            # Update statistics
            self._stats.episode_rewards[self._steps] += reward
            self._stats.episode_lengths[self._steps] = t
            self._stats.episode_times[self._steps] = time.clock() - start_time

            # TD Update
            best_next_action = np.argmax(self._Q[next_state])
            td_target = reward + self._discount_factor * self._Q[next_state, best_next_action]
            td_delta = td_target - self._Q[state, action]
            self._stats.episode_deltas[self._steps] = td_delta
            self._Q[state, action] += self._alpha * td_delta

            # Decay epsilon
            self._epsilon -= self._epsilon * self._epsilon_decay

            total_reward += reward
            self._last_delta = max(self._last_delta, td_delta)

            episode_steps += 1
            if done:
                break

            state = next_state

        if self._last_delta < self._theta:
            self._consecutive_sub_theta_episodes += 1
        else:
            self._consecutive_sub_theta_episodes = 0

        self._step_times.append(time.clock() - start_time)

        self._steps += 1

        return self.get_policy(), self.get_value(), self._steps, self._step_times[-1], \
            total_reward/episode_steps, self._last_delta, self.has_converged()

    def reset(self):
        self._init_q()
        self._steps = 0
        self._step_times = []
        self._last_delta = 0
        self._epsilon = self._initial_epsilon
        self._stats = EpisodeStats(self._max_episodes)
        self._consecutive_sub_theta_episodes = 0

    def has_converged(self):
        return (self._steps >= self._min_episodes and
                self._consecutive_sub_theta_episodes >= self._min_consecutive_sub_theta_episodes) \
               or self._steps > self._max_episodes

    def get_convergence(self):
        return self._last_delta

    def run_until_converged(self):
        while not self.has_converged():
            self.step()

    def get_environment(self):
        return self._env

    def get_stats(self):
        return self._stats

    def get_q(self):
        return self._Q

    def get_policy(self):
        policy = np.zeros([self._env.nS, self._env.nA])
        for s in range(self._env.nS):
            best_action = np.argmax(self._Q[s])
            # Always take the best action
            policy[s, best_action] = 1.0

        return policy

    def get_value(self):
        v = np.zeros(self._env.nS)
        for s in range(self._env.nS):
            v[s] = np.max(self._Q[s])

        return v

    def _init_q(self):
        if self._q_init == 'random':
            self._Q = np.random.rand(self._env.observation_space.n, self._env.action_space.n)/1000.0
        elif int(self._q_init) == 0:
            self._Q = np.zeros(shape=(self._env.observation_space.n, self._env.action_space.n))
        else:
            self._Q = np.full((self._env.observation_space.n, self._env.action_space.n), float(self._q_init))

    def _policy_function(self, observation):
        A = np.ones(self._env.action_space.n, dtype=float) * self._epsilon / self._env.action_space.n
        best_action = np.argmax(self._Q[observation])
        A[best_action] += (1.0 - self._epsilon)
        return A

    # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        def policy_fn(observation):
            A = np.ones(self._env.action_space.n, dtype=float) * self._epsilon / self._env.action_space.n
            best_action = np.argmax(self._Q[observation])
            A[best_action] += (1.0 - self._epsilon)
            return A
        return policy_fn
