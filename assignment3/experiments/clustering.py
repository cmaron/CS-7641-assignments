import os
from functools import reduce

import pandas as pd
import numpy as np

from collections import Counter
from sklearn.manifold import TSNE
from time import clock
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score, \
    silhouette_samples as sil_samples
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM

import experiments


def cluster_acc(y, cluster_labels):
    assert (y.shape == cluster_labels.shape)
    pred = np.empty_like(y)
    for label in set(cluster_labels):
        mask = cluster_labels == label
        sub = y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)
    return acc(y, pred)


class CustomGMM(GMM):
    def transform(self, x):
        return self.predict_proba(x)


class LabelKMeans(kmeans):
    def transform(self, x):
        trans = self.transform(x)
        trans['cluster'] = trans.labels_

        return trans


class ClusteringExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose
        self._nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100)]
        self._nn_reg = [10 ** -x for x in range(1, 5)]
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        self._old_out = None

    def experiment_name(self):
        return 'clustering'

    def perform(self):
        return self.__do_perform()

    def perform_for_subexperiment(self, custom_out, main_experiment):
        return self.__do_perform(custom_out=custom_out, main_experiment=main_experiment)

    # The custom_out and main_experiment are used for experiment that need to call this as part of a larger
    # experiment
    def __do_perform(self, custom_out=None, main_experiment=None):
        if custom_out is not None:
            # if not os.path.exists(custom_out):
            #     os.makedirs(custom_out)
            self._old_out = self._out
            self._out = custom_out
        elif self._old_out is not None:
            self._out = self._old_out

        if main_experiment is not None:
            self.log("Performing {} as part of {}".format(self.experiment_name(), main_experiment.experiment_name()))
        else:
            self.log("Performing {}".format(self.experiment_name()))

        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/clustering.py
        # %% Data for 1-3
        sse = defaultdict(list)
        ll = defaultdict(list)
        bic = defaultdict(list)
        sil = defaultdict(lambda: defaultdict(list))
        sil_s = np.empty(shape=(2*len(self._clusters)*self._details.ds.training_x.shape[0],4), dtype='<U21')
        acc = defaultdict(lambda: defaultdict(float))
        adj_mi = defaultdict(lambda: defaultdict(float))
        km = kmeans(random_state=self._details.seed)
        gmm = GMM(random_state=self._details.seed)

        st = clock()
        j = 0
        for k in self._clusters:
            km.set_params(n_clusters=k)
            gmm.set_params(n_components=k)
            km.fit(self._details.ds.training_x)
            gmm.fit(self._details.ds.training_x)

            km_labels = km.predict(self._details.ds.training_x)
            gmm_labels = gmm.predict(self._details.ds.training_x)

            sil[k]['Kmeans'] = sil_score(self._details.ds.training_x, km_labels)
            sil[k]['GMM'] = sil_score(self._details.ds.training_x, gmm_labels)

            km_sil_samples = sil_samples(self._details.ds.training_x, km_labels)
            gmm_sil_samples = sil_samples(self._details.ds.training_x, gmm_labels)
            # There has got to be a better way to do this, but I can't brain right now
            for i, x in enumerate(km_sil_samples):
                sil_s[j] = [k, 'Kmeans', round(x, 6), km_labels[i]]
                j += 1
            for i, x in enumerate(gmm_sil_samples):
                sil_s[j] = [k, 'GMM', round(x, 6), gmm_labels[i]]
                j += 1

            sse[k] = [km.score(self._details.ds.training_x)]
            ll[k] = [gmm.score(self._details.ds.training_x)]
            bic[k] = [gmm.bic(self._details.ds.training_x)]

            acc[k]['Kmeans'] = cluster_acc(self._details.ds.training_y, km_labels)
            acc[k]['GMM'] = cluster_acc(self._details.ds.training_y, gmm_labels)

            adj_mi[k]['Kmeans'] = ami(self._details.ds.training_y, km_labels)
            adj_mi[k]['GMM'] = ami(self._details.ds.training_y, gmm_labels)

            self.log("Cluster: {}, time: {}".format(k, clock() - st))

        sse = (-pd.DataFrame(sse)).T
        sse.index.name = 'k'
        sse.columns = ['{} sse (left)'.format(self._details.ds_readable_name)]

        ll = pd.DataFrame(ll).T
        ll.index.name = 'k'
        ll.columns = ['{} log-likelihood'.format(self._details.ds_readable_name)]

        bic = pd.DataFrame(bic).T
        bic.index.name = 'k'
        bic.columns = ['{} BIC'.format(self._details.ds_readable_name)]

        sil = pd.DataFrame(sil).T
        sil_s = pd.DataFrame(sil_s, columns=['k', 'type', 'score', 'label']).set_index('k')  #.T
        # sil_s = sil_s.T
        acc = pd.DataFrame(acc).T
        adj_mi = pd.DataFrame(adj_mi).T

        sil.index.name = 'k'
        sil_s.index.name = 'k'
        acc.index.name = 'k'
        adj_mi.index.name = 'k'

        sse.to_csv(self._out.format('{}_sse.csv'.format(self._details.ds_name)))
        ll.to_csv(self._out.format('{}_logliklihood.csv'.format(self._details.ds_name)))
        bic.to_csv(self._out.format('{}_bic.csv'.format(self._details.ds_name)))
        sil.to_csv(self._out.format('{}_sil_score.csv'.format(self._details.ds_name)))
        sil_s.to_csv(self._out.format('{}_sil_samples.csv'.format(self._details.ds_name)))
        acc.to_csv(self._out.format('{}_acc.csv'.format(self._details.ds_name)))
        adj_mi.to_csv(self._out.format('{}_adj_mi.csv'.format(self._details.ds_name)))

        # %% NN fit data (2,3)
        grid = {'km__n_clusters': self._clusters, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        km = kmeans(random_state=self._details.seed, n_jobs=self._details.threads)
        pipe = Pipeline([('km', km), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, grid, type='kmeans')
        self.log("KMmeans Grid search complete")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_cluster_kmeans.csv'.format(self._details.ds_name)))

        grid = {'gmm__n_components': self._clusters, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        gmm = CustomGMM(random_state=self._details.seed)
        pipe = Pipeline([('gmm', gmm), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, grid, type='gmm')
        self.log("GMM search complete")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_cluster_GMM.csv'.format(self._details.ds_name)))

        # %% For chart 4/5
        self._details.ds.training_x2D = TSNE(verbose=10, random_state=self._details.seed).fit_transform(
            self._details.ds.training_x
        )

        ds_2d = pd.DataFrame(np.hstack((self._details.ds.training_x2D, np.atleast_2d(self._details.ds.training_y).T)),
                             columns=['x', 'y', 'target'])
        ds_2d.to_csv(self._out.format('{}_2D.csv'.format(self._details.ds_name)))
        self.log("Done")

    def perform_cluster(self, dim_param):
        self.log('Clustering for a specific dim is not run for {}'.format(self.experiment_name()))
