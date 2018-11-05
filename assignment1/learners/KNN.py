from sklearn import neighbors

import learners


class KNNLearner(learners.BaseLearner):
    def __init__(self,
                 verbose=False,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=1,
                 **kwargs):
        super().__init__(verbose)
        self._learner = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs)

    def learner(self):
        return self._learner
