import numpy as np
from sklearn import tree
import sklearn.model_selection as ms

import learners


class DTLearner(learners.BaseLearner):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False,
                 verbose=False):
        super().__init__(verbose)
        self._learner = tree.DecisionTreeClassifier(
                 criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 class_weight=class_weight,
                 presort=presort)

    def learner(self):
        return self._learner

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_

    def fit(self, training, testing, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        return self._learner.fit(training, testing, sample_weight=sample_weight, check_input=check_input,
                                 X_idx_sorted=X_idx_sorted)

    def write_visualization(self, path):
        """
        Write a visualization of the given learner to the given path (including file name but not extension)
        :return: self
        """
        return tree.export_graphviz(self._learner, out_file='{}.dot'.format(path))
