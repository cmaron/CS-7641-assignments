import warnings

import numpy as np
import sklearn

import experiments
import learners


class KNNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/KNN.py
        params = {'KNN__metric': ['manhattan', 'euclidean', 'chebyshev'], 'KNN__n_neighbors': np.arange(1, 51, 3),
                  'KNN__weights': ['uniform', 'distance']}
        complexity_param = {'name': 'KNN__n_neighbors', 'display_name': 'Neighbor count', 'values': np.arange(1, 51, 1)}

        learner = learners.KNNLearner(n_jobs=self._details.threads)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'KNN', 'KNN',
                                       params, complexity_param=complexity_param,
                                       seed=self._details.seed, threads=self._details.threads,
                                       verbose=self._verbose)
