from sklearn import ensemble

import learners


class BoostingLearner(learners.BaseLearner):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 verbose=False):
        super().__init__(verbose)
        self._learner = ensemble.AdaBoostClassifier(
                 base_estimator=base_estimator,
                 n_estimators=n_estimators,
                 learning_rate=learning_rate,
                 algorithm=algorithm,
                 random_state=random_state)

    def learner(self):
        return self._learner
