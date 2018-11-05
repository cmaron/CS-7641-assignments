from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np
import scipy.sparse as sps

from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from scipy.linalg import pinv

import experiments


def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def reconstruction_error(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p@w)@(x.T)).T  # Unproject projected data
    errors = np.square(x-reconstructed)
    return np.nanmean(errors)


# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
        self.model = model
        self.n = n

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X):
        return X[:, self.model.feature_importances_.argsort()[::-1][:self.n]]


class RPExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose
        self._nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100)]
        self._nn_reg = [10 ** -x for x in range(1, 5)]
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        self._dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    def experiment_name(self):
        return 'RP'

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/RP.py
        self.log("Performing {}".format(self.experiment_name()))

        # TODO: Use a diff random state? Might be ok as-is
        # %% Data for 1
        tmp = defaultdict(dict)
        for i, dim in product(range(10), self._dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            tmp[dim][i] = pairwise_dist_corr(rp.fit_transform(self._details.ds.training_x), self._details.ds.training_x)
        tmp = pd.DataFrame(tmp).T
        tmp.to_csv(self._out.format('{}_scree1.csv'.format(self._details.ds_name)))

        tmp = defaultdict(dict)
        for i, dim in product(range(10), self._dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            rp.fit(self._details.ds.training_x)
            tmp[dim][i] = reconstruction_error(rp, self._details.ds.training_x)
        tmp = pd.DataFrame(tmp).T
        tmp.to_csv(self._out.format('{}_scree2.csv'.format(self._details.ds_name)))

        # %% Data for 2
        grid = {'rp__n_components': self._dims, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        rp = SparseRandomProjection(random_state=self._details.seed)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('rp', rp), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, final_estimator = self.gs_with_best_estimator(pipe, grid)
        self.log("Grid search complete")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_dim_red.csv'.format(self._details.ds_name)))
        self.log("Done")

    def perform_cluster(self, dim_param):
        self.log('Running clustering for {} with dim param {}'.format(self.experiment_name(), dim_param))

        # TODO: USE UNSUPERVISED METHOD TO GET THIS BEST VALUE
        # %% Data for 3
        # Set this from chart 2 and dump, use clustering script to finish up
        rp = SparseRandomProjection(n_components=dim_param, random_state=self._details.seed)

        # ANN based on best params from assignment 1
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('rp', rp), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, experiments.BEST_NN_PARAMS, type='ass1')

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_ass1_dim_red.csv'.format(self._details.ds_name)))

        hdf_path = self.dump_for_clustering(lambda x: rp.fit_transform(x.get_details().ds.training_x))

        # Run clustering as a subexperiment
        self.log("Running clustering sub-experiment")
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)
        experiments.run_subexperiment(self, self._out.format('clustering/'), updated_ds)
