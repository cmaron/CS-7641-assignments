import warnings

import numpy as np
import sklearn

import experiments
import learners


class BoostingExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/Boosting.py
        # Search for good alphas
        alphas = np.arange(1, 11)

        max_depths = np.arange(1, 41, 1)  # np.arange(1, 11)
        base = learners.DTLearner(criterion='gini', class_weight='balanced', random_state=self._details.seed)
        of_base = learners.DTLearner(criterion='gini', class_weight='balanced', random_state=self._details.seed)

        booster = learners.BoostingLearner(algorithm='SAMME', learning_rate=1, base_estimator=base,
                                           random_state=self._details.seed)
        of_booster = learners.BoostingLearner(algorithm='SAMME', learning_rate=1, base_estimator=of_base,
                                              random_state=self._details.seed)

        # TODO: No 90 here?
        params = {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 100],
                  'Boost__base_estimator__max_depth': max_depths}
        iteration_params = {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        of_params = {'Boost__base_estimator__max_depth': 100, 'Boost__n_estimators': 50}
        complexity_param = {'name': 'Boost__n_estimators', 'display_name': 'Estimator count', 'x_scale': 'log',
                            'values': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, booster,
                                       'Boost', 'Boost', params, complexity_param=complexity_param,
                                       seed=self._details.seed, threads=self._details.threads, verbose=self._verbose)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       of_booster, 'Boost_OF', 'Boost', of_params, seed=self._details.seed,
                                       iteration_params=iteration_params, threads=self._details.threads,
                                       verbose=self._verbose, iteration_lc_only=True)
