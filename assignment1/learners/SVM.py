import numpy as np

from sklearn import svm
from sklearn.linear_model import stochastic_gradient, SGDClassifier
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

import learners

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
class SVMLearner(learners.BaseLearner):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        super().__init__(verbose)
        self._learner = svm.SVC(C=C,
                                kernel=kernel,
                                degree=degree,
                                gamma=gamma,
                                coef0=coef0,
                                shrinking=shrinking,
                                probability=probability,
                                tol=tol,
                                cache_size=cache_size,
                                class_weight=class_weight,
                                verbose=verbose,
                                max_iter=max_iter,
                                decision_function_shape=decision_function_shape,
                                random_state=random_state)

    def learner(self):
        return self._learner


class LinearSVMLearner(learners.BaseLearner):
    def __init__(self, C=1.0, loss='squared_hinge', dual=True, penalty='l2',
                 multi_class='ovr', intercept_scaling=1, fit_intercept=True,
                 tol=1e-3, class_weight=None,
                 verbose=False, max_iter=-1, random_state=None):
        super().__init__(verbose)
        self._learner = svm.LinearSVC(penalty=penalty,
                                      loss=loss,
                                      dual=dual,
                                      tol=tol,
                                      C=C,
                                      multi_class=multi_class,
                                      fit_intercept=fit_intercept,
                                      intercept_scaling=intercept_scaling,
                                      class_weight=class_weight,
                                      verbose=verbose,
                                      random_state=random_state,
                                      max_iter=max_iter)

    def learner(self):
        return self._learner


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
class RBFSVMLearner(learners.BaseLearner):
    def __init__(self,
                 loss="hinge",
                 penalty='l2',
                 alpha=1e-9,
                 l1_ratio=0,
                 fit_intercept=True,
                 max_iter=None,
                 tol=None,
                 shuffle=True,
                 verbose=False,
                 epsilon=stochastic_gradient.DEFAULT_EPSILON,
                 n_jobs=1,
                 random_state=None,
                 learning_rate="optimal",
                 eta0=0.0,
                 power_t=0.5,
                 class_weight=None,
                 warm_start=False,
                 average=False,
                 n_iter=2000,
                 gamma_frac=0.1,
                 use_linear=False):
        super().__init__(verbose)
        self._alpha = alpha
        self._gamma_frac = gamma_frac
        self._n_iter = n_iter
        self._use_linear = use_linear
        self._learner = SGDClassifier(loss=loss,
                                      penalty=penalty,
                                      alpha=self._alpha,
                                      l1_ratio=l1_ratio,
                                      fit_intercept=fit_intercept,
                                      max_iter=max_iter,
                                      tol=tol,
                                      shuffle=shuffle,
                                      verbose=verbose,
                                      epsilon=epsilon,
                                      n_jobs=n_jobs,
                                      average=average,
                                      learning_rate=learning_rate,
                                      eta0=eta0,
                                      power_t=power_t,
                                      class_weight=class_weight,
                                      warm_start=warm_start,
                                      n_iter=self._n_iter,
                                      random_state=random_state)

        self.gamma = None
        self.X_ = None
        self.classes_ = None
        self.kernels_ = None
        self.y_ = None

    def learner(self):
        return self._learner

    def fit(self, training_data, classes):
        if self._use_linear:
            return self._learner.fit(training_data, classes)
        # Check that training_data, classes
        training_data, classes = check_X_y(training_data, classes)

        # Get the kernel matrix
        dist = euclidean_distances(training_data, squared=True)
        median = np.median(dist)
        del dist
        gamma = median
        gamma *= self._gamma_frac
        self.gamma = 1 / gamma
        kernels = rbf_kernel(training_data, None, self.gamma)

        self.X_ = training_data
        self.classes_ = unique_labels(classes)
        self.kernels_ = kernels
        self.y_ = classes
        self._learner.fit(self.kernels_, self.y_)

        # Return the classifier
        return self

    def predict(self, data):
        if self._use_linear:
            return self._learner.predict(data)

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', '_learner', 'kernels_'])
        # Input validation
        data = check_array(data)
        new_kernels = rbf_kernel(data, self.X_, self.gamma)
        pred = self._learner.predict(new_kernels)
        return pred

    # We pass gamma_frac around
    def get_params(self, deep=True):
        """
        Get the current parameters for the learner. This passes the call back to the learner from learner()

        :param deep: If true, fetch deeply
        :return: The parameters
        """
        extra_params = {'gamma_frac': self._gamma_frac, 'use_linear': self._use_linear}
        params = self._learner.get_params(deep)

        return {k: v for d in (params, extra_params) for k, v in d.items()}

    def set_params(self, **params):
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        if 'gamma_frac' in params:
            self._gamma_frac = params.pop('gamma_frac', None)
        if 'use_linear' in params:
            self._use_linear = params.pop('use_linear', None)

        return self._learner.set_params(**params)
