import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import experiments


class PCAExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose
        self._nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100)]
        self._nn_reg = [10 ** -x for x in range(1, 5)]
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        self._dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    def experiment_name(self):
        return 'PCA'

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/PCA.py
        self.log("Performing {}".format(self.experiment_name()))

        # %% Data for 1
        pca = PCA(random_state=self._details.seed)
        pca.fit(self._details.ds.training_x)
        tmp = pd.Series(data=pca.explained_variance_, index=range(1, min(pca.explained_variance_.shape[0], 500) + 1))
        tmp.to_csv(self._out.format('{}_scree.csv'.format(self._details.ds_name)))

        # If the ds is small or the number of components is too large, the full solver is used for PCA and as a result
        # we need to re-create the array of dimensions. In that case we'll create a linear distribution from 2 to
        # ds.shape[1]
        if (max(self._details.ds.training_x.shape) <= 500 or
           self._details.ds.training_x.shape[1] > (0.8 * min(self._details.ds.training_x.shape))):
            self._dims = list(set(np.linspace(2, self._details.ds.training_x.shape[1], num=len(self._dims), dtype=int)))
            self.log("Must use full solver, new dims are {}".format(self._dims))

        # %% Data for 2
        grid = {'pca__n_components': self._dims, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        pca = PCA(random_state=self._details.seed)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('pca', pca), ('NN', mlp)], memory=experiments.pipeline_memory)
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
        pca = PCA(n_components=dim_param, random_state=self._details.seed)

        # ANN based on best params from assignment 1
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('pca', pca), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, experiments.BEST_NN_PARAMS, type='ass1')

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_ass1_dim_red.csv'.format(self._details.ds_name)))

        hdf_path = self.dump_for_clustering(lambda x: pca.fit_transform(x.get_details().ds.training_x))

        # Run clustering as a subexperiment
        self.log("Running clustering sub-experiment")
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)
        experiments.run_subexperiment(self, self._out.format('clustering/'), updated_ds)
