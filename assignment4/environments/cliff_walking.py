import numpy as np
import sys
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# Adapted from https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
class WindyCliffWalkingEnv(discrete.DiscreteEnv):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.

    The cliff is windy, however, so the agent is sometime pushed down

    Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/the-book-2nd.html

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal  (earning 100 pts in the process).
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, wind_prob=0.1):
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)
        self.wind_prob = wind_prob

        self.desc = np.asarray([
            "RRRRRRRRRRRR",
            "RRRRRRRRRRRR",
            "RRRRRRRRRRRR",
            "SCCCCCCCCCCG",
        ], dtype='c')

        nS = np.prod(self.shape)
        nA = 4

        # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/windy_gridworld.py
        # Wind strength
        winds = np.zeros(self.shape)
        winds[:, [3, 4, 5, 8]] = 1 * np.random.uniform(0.0, 1.0)
        winds[:, [6, 7]] = 2 * np.random.uniform(0.0, 1.0)

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(WindyCliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        # # IF the wind is blowing, move the agent down
        # wind_blows = np.random.uniform(0.0, 1.0) <= self.wind_prob
        # if wind_blows:
        #     new_position = np.array(current) + np.array([1, 0])
        # else:
        #     new_position = np.array(current) + np.array(delta)

        new_position = np.array(current) + np.array(delta) + (np.array([1, 0]) * winds[tuple(current)])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == terminal_state
        if is_done:
            return [(1.0, new_state, 100, is_done)]
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // 12, self.s % 4
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Up", "Right", "Down", "Left"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):
        return {
            b'S': 'green',
            b'R': 'lightslategray',
            b'C': 'black',
            b'G': 'gold',
        }

    def directions(self):
        return {
            0: '⬆',
            1: '➡',
            2: '⬇',
            3: '⬅'
        }

    def new_instance(self):
        return WindyCliffWalkingEnv(wind_prob=self.wind_prob)

