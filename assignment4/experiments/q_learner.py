import json
import os
import time
import numpy as np

from .base import BaseExperiment, OUTPUT_DIRECTORY

import solvers

if not os.path.exists(OUTPUT_DIRECTORY + '/Q'):
    os.makedirs(OUTPUT_DIRECTORY + '/Q')
if not os.path.exists(OUTPUT_DIRECTORY + '/Q/pkl'):
    os.makedirs(OUTPUT_DIRECTORY + '/Q/pkl')
if not os.path.exists(OUTPUT_DIRECTORY + '/images/Q'):
    os.makedirs(OUTPUT_DIRECTORY + '/images/Q')


class QLearnerExperiment(BaseExperiment):
    def __init__(self, details, verbose=False):
        self.max_episodes = 2000

        super(QLearnerExperiment, self).__init__(details, verbose)

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Q-Learner
        self._details.env.reset()
        map_desc = self._details.env.unwrapped.desc

        grid_file_name = '{}/Q/{}_grid.csv'.format(OUTPUT_DIRECTORY, self._details.env_name)
        with open(grid_file_name, 'w') as f:
            f.write("params,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        alphas = [0.1, 0.5, 0.9]
        q_inits = ['random', 0]
        epsilons = [0.1, 0.3, 0.5]
        epsilon_decays = [0.0001]
        discount_factors = np.round(np.linspace(0, 0.9, num=10), 2)
        dims = len(discount_factors) * len(alphas) * len(q_inits) * len(epsilons) * len(epsilon_decays)
        self.log("Searching Q in {} dimensions".format(dims))

        runs = 1
        for alpha in alphas:
            for q_init in q_inits:
                for epsilon in epsilons:
                    for epsilon_decay in epsilon_decays:
                        for discount_factor in discount_factors:
                            t = time.clock()
                            self.log("{}/{} Processing Q with alpha {}, q_init {}, epsilon {}, epsilon_decay {},"
                                     " discount_factor {}".format(
                                runs, dims, alpha, q_init, epsilon, epsilon_decay, discount_factor
                            ))

                            qs = solvers.QLearningSolver(self._details.env, self.max_episodes,
                                                         discount_factor=discount_factor,
                                                         alpha=alpha,
                                                         epsilon=epsilon, epsilon_decay=epsilon_decay,
                                                         q_init=q_init, verbose=self._verbose)

                            stats = self.run_solver_and_collect(qs, self.convergence_check_fn)

                            self.log("Took {} episodes".format(len(stats.steps)))
                            stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}.csv'.format(OUTPUT_DIRECTORY, self._details.env_name,
                                                                          alpha, q_init, epsilon, epsilon_decay,
                                                                          discount_factor))
                            stats.pickle_results('{}/Q/pkl/{}_{}_{}_{}_{}_{}_{}.pkl'.format(OUTPUT_DIRECTORY,
                                                                                            self._details.env_name,
                                                                                            alpha, q_init, epsilon,
                                                                                            epsilon_decay,
                                                                                            discount_factor,
                                                                                            '{}'), map_desc.shape,
                                                  step_size=self.max_episodes/20.0)
                            stats.plot_policies_on_map('{}/images/Q/{}_{}_{}_{}_{}_{}_{}.png'.format(OUTPUT_DIRECTORY,
                                                                                                  self._details.env_name,
                                                                                                  alpha, q_init, epsilon,
                                                                                                  epsilon_decay,
                                                                                                  discount_factor,
                                                                                                  '{}_{}'),
                                                       map_desc, self._details.env.colors(),
                                                       self._details.env.directions(),
                                                       'Q-Learner', 'Episode', self._details,
                                                       step_size=self.max_episodes / 20.0,
                                                       only_last=True)

                            # We have extra stats about the episode we might want to look at later
                            episode_stats = qs.get_stats()
                            episode_stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}_episode.csv'.format(OUTPUT_DIRECTORY,
                                                                                             self._details.env_name,
                                                                                             alpha, q_init, epsilon,
                                                                                             epsilon_decay,
                                                                                             discount_factor))

                            optimal_policy_stats = self.run_policy_and_collect(qs, stats.optimal_policy)
                            self.log('{}'.format(optimal_policy_stats))
                            optimal_policy_stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}_optimal.csv'.format(OUTPUT_DIRECTORY,
                                                                                                 self._details.env_name,
                                                                                                 alpha, q_init, epsilon,
                                                                                                 epsilon_decay,
                                                                                                 discount_factor))

                            with open(grid_file_name, 'a') as f:
                                f.write('"{}",{},{},{},{},{},{},{}\n'.format(
                                    json.dumps({
                                        'alpha': alpha,
                                        'q_init': q_init,
                                        'epsilon': epsilon,
                                        'epsilon_decay': epsilon_decay,
                                        'discount_factor': discount_factor,
                                    }).replace('"', '""'),
                                    time.clock() - t,
                                    len(optimal_policy_stats.rewards),
                                    optimal_policy_stats.reward_mean,
                                    optimal_policy_stats.reward_median,
                                    optimal_policy_stats.reward_min,
                                    optimal_policy_stats.reward_max,
                                    optimal_policy_stats.reward_std,
                                ))
                            runs += 1
