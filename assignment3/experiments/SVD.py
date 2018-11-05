import os

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import experiments


class SVDExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose
        self._nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100)]
        self._nn_reg = [10 ** -x for x in range(1, 5)]
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        self._dims = np.arange(1, details.ds.features.shape[1])

    def experiment_name(self):
        return 'SVD'

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/RF.py
        self.log("Performing {}".format(self.experiment_name()))

        # %% Data for 1
        svd = TruncatedSVD(n_components=self._details.ds.features.shape[1]-1, random_state=self._details.seed)
        svd.fit(self._details.ds.training_x)
        tmp = pd.Series(data=svd.explained_variance_, index=range(1, min(svd.explained_variance_.shape[0], 500) + 1))
        tmp.to_csv(self._out.format('{}_scree.csv'.format(self._details.ds_name)))

        # # %% Data for 2
        svd = TruncatedSVD(random_state=self._details.seed)
        grid = {'svd__n_components': self._dims, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('svd', svd), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, final_estimator = self.gs_with_best_estimator(pipe, grid)
        self.log("Grid search complete")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_dim_red.csv'.format(self._details.ds_name)))
        self.log("Done")

    def perform_cluster(self, dim_param):
        self.log('Running clustering for {} with dim param {}'.format(self.experiment_name(), dim_param))

        # # TODO: USE UNSUPERVISED METHOD TO GET THIS BEST VALUE
        # # %% Data for 3
        # # Set this from chart 2 and dump, use clustering script to finish up
        # Set this from chart 2 and dump, use clustering script to finish up
        svd = TruncatedSVD(n_components=dim_param, random_state=self._details.seed)

        # ANN based on best params from assignment 1
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('svd', svd), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, experiments.BEST_NN_PARAMS, type='ass1')

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_ass1_dim_red.csv'.format(self._details.ds_name)))

        hdf_path = self.dump_for_clustering(lambda x: svd.fit_transform(x.get_details().ds.training_x))

        # Run clustering as a subexperiment
        self.log("Running clustering sub-experiment")
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)
        experiments.run_subexperiment(self, self._out.format('clustering/'), updated_ds)
