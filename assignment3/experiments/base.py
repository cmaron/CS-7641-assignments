import logging
import os
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV

from .scoring import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))


class ExperimentDetails(object):
    def __init__(self, ds, ds_name, ds_readable_name, best_nn_params, threads, seed):
        self.ds = ds
        self.ds_name = ds_name
        self.ds_readable_name = ds_readable_name
        self.best_nn_params = best_nn_params
        self.threads = threads
        self.seed = seed

    def __str__(self):
        return 'ExperimentDetails(ds={}, ds_name={}, ds_readable_name={}, best_nn_params={}, threads={}, seed={})'.format(
            self.ds,
            self.ds_name,
            self.ds_readable_name,
            self.best_nn_params,
            self.threads,
            self.seed
        )


class BaseExperiment(ABC):
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

        out = '{}/{}'.format(OUTPUT_DIRECTORY, self.experiment_name())
        if not os.path.exists(out):
            os.makedirs(out)
        self._out = '{}/{}'.format(out, '{}')

        self._scorer, _ = get_scorer(self._details.ds)

    def get_details(self):
        return self._details

    def get_vebose(self):
        return self._verbose

    @abstractmethod
    def experiment_name(self):
        pass

    @abstractmethod
    def perform(self):
        pass

    @abstractmethod
    def perform_cluster(self, dim_param):
        pass

    def gs_with_best_estimator(self, pipe, grid, type=None):
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5, scoring=self._scorer, n_jobs=self._details.threads)
        gs.fit(self._details.ds.training_x, self._details.ds.training_y)

        best_estimator = gs.best_estimator_.fit(self._details.ds.training_x, self._details.ds.training_y)
        final_estimator = best_estimator._final_estimator
        best_params = pd.DataFrame([best_estimator.get_params()])
        final_estimator_params = pd.DataFrame([final_estimator.get_params()])
        if type:
            best_params.to_csv(self._out.format('{}_{}_best_params.csv'.format(type, self._details.ds_name)),
                               index=False)
            final_estimator_params.to_csv(
                self._out.format('{}_{}_final_estimator_params.csv'.format(type, self._details.ds_name)),
                index=False
            )
        else:
            best_params.to_csv(self._out.format('{}_best_params.csv'.format(self._details.ds_name)),
                               index=False)
            final_estimator_params.to_csv(
                self._out.format('{}_final_estimator_params.csv'.format(self._details.ds_name)),
                index=False
            )

        return gs, best_estimator  #, final_estimator

    def dump_for_clustering(self, learning_func):
        hdf_path = self._out.format('datasets.hdf')
        ds_features = learning_func(self)
        ds_2 = pd.DataFrame(np.hstack((ds_features, np.atleast_2d(self._details.ds.training_y).T)))
        cols = list(range(ds_2.shape[1]))
        cols[-1] = 'Class'
        ds_2.columns = cols
        ds_2.to_hdf(hdf_path, self._details.ds_name, complib='blosc', complevel=9)
        return hdf_path

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))
