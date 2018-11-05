import numpy as np

import experiments
import learners


class DTExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/DT.py
        max_depths = np.arange(1, 41, 1)
        params = {'DT__criterion': ['gini', 'entropy'], 'DT__max_depth': max_depths,
                  'DT__class_weight': ['balanced']}
        complexity_param = {'name': 'DT__max_depth', 'display_name': 'Max Depth', 'values': max_depths}

        learner = learners.DTLearner(random_state=self._details.seed)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'DT', 'DT', params,
                                       complexity_param=complexity_param, seed=self._details.seed,
                                       threads=self._details.threads,
                                       verbose=self._verbose)
