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
        max_depths = np.arange(1, 11, 1)

        # NOTE: Criterion may need to be adjusted here depending on the dataset
        base = learners.DTLearner(criterion='entropy', class_weight='balanced', max_depth=10,
                                  random_state=self._details.seed)
        of_base = learners.DTLearner(criterion='entropy', class_weight='balanced', random_state=self._details.seed)

        booster = learners.BoostingLearner(algorithm='SAMME', learning_rate=1, base_estimator=base,
                                           random_state=self._details.seed)
        of_booster = learners.BoostingLearner(algorithm='SAMME', learning_rate=1, base_estimator=of_base,
                                              random_state=self._details.seed)

        params = {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 90, 100],
                  'Boost__learning_rate': [(2**x)/100 for x in range(7)]+[1],
                  'Boost__base_estimator__max_depth': max_depths}
        iteration_details = {
            'params': {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        }
        of_params = {'Boost__base_estimator__max_depth': None}
        complexity_param = {'name': 'Boost__learning_rate', 'display_name': 'Learning rate', 'x_scale': 'log',
                            'values': [(2**x)/100 for x in range(7)]+[1]}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params = {'base_estimator__max_depth': 8, 'learning_rate': 0.32, 'n_estimators': 90}
        #
        # Dataset 2:
        # best_params = {'base_estimator__max_depth': 6, 'learning_rate': 0.16, 'n_estimators': 20}

        if best_params is not None:
            booster.set_params(**best_params)
            of_booster.set_params(**best_params)

        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, booster,
                                       'Boost', 'Boost', params, complexity_param=complexity_param,
                                       iteration_details=iteration_details, best_params=best_params,
                                       seed=self._details.seed, threads=self._details.threads, verbose=self._verbose)

        # TODO: This should turn OFF regularization
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       of_booster, 'Boost_OF', 'Boost', of_params, seed=self._details.seed,
                                       iteration_details=iteration_details,
                                       best_params=best_params, threads=self._details.threads,
                                       verbose=self._verbose, iteration_lc_only=True)
