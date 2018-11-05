import warnings

import numpy as np
import sklearn

import experiments
import learners


class SVMExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
        alphas = [10 ** -x for x in np.arange(1, 9.01, 1 / 2)]

        samples = self._details.ds.features.shape[0]

        gamma_fracs = np.arange(0.2, 2.1, 0.2)

        params = {'SVM__alpha': alphas, 'SVM__max_iter': [int((1e6/samples)/.8)+1], 'SVM__gamma_frac': gamma_fracs}
        complexity_param = {'name': 'SVM__gamma_frac', 'display_name': 'Gamma Fraction', 'values': gamma_fracs}

        iteration_params = {'SVM__max_iter': [2**x for x in range(12)]}

        learner = learners.SVMLearner(tol=None)
        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVM_RBF', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_params=iteration_params,
            threads=self._details.threads, verbose=self._verbose)

        of_params = best_params.copy()
        of_params['SVM__alpha'] = 1e-16
        learner = learners.SVMLearner(n_jobs=self._details.threads)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'SVM_RBF_OF', 'SVM', of_params, seed=self._details.seed,
                                       iteration_params=iteration_params,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)

