import numpy as np

import experiments
import learners


class ANNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/ANN.py
        # Search for good alphas
        alphas = [10 ** -x for x in np.arange(-1, 9.01, 1 / 2)]

        # TODO: Allow for tuning of hidden layers based on dataset provided
        d = self._details.ds.features.shape[1]
        hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]

        params = {'MLP__activation': ['relu', 'logistic'], 'MLP__alpha': alphas,
                  'MLP__hidden_layer_sizes': hiddens}

        timing_params = {'MLP__early_stopping': False}
        iteration_params = {'MLP__max_iter':
                            [2 ** x for x in range(12)] + [2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                                                           3000]}
        complexity_param = {'name': 'MLP__alpha', 'display_name': 'Alpha', 'x_scale': 'log',
                            'values': alphas}

        learner = learners.ANNLearner(tol=1e-8, verbose=self._verbose)
        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'ANN', 'MLP',
            params,
            complexity_param=complexity_param,
            seed=self._details.seed,
            timing_params=timing_params,
            iteration_pipe_params=timing_params, iteration_params=iteration_params,
            threads=self._details.threads, verbose=self._verbose)

        of_params = best_params.copy()
        of_params['MLP__alpha'] = 0
        learner = learners.ANNLearner()
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'ANN_OF', 'MLP', of_params, seed=self._details.seed, timing_params=timing_params,
                                       iteration_pipe_params=timing_params, iteration_params=iteration_params,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)
