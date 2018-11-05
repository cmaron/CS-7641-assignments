from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseLearner(ABC, BaseEstimator, ClassifierMixin):
    """
    Base learner for classification-based learning
    """
    def __init__(self, verbose):
        self._verbose = verbose

    @property
    @abstractmethod
    def learner(self):
        pass

    def get_params(self, deep=True):
        """
        Get the current parameters for the learner. This passes the call back to the learner from learner()

        :param deep: If true, fetch deeply
        :return: The parameters
        """
        return self.learner().get_params(deep)

    def set_params(self, **params):
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        return self.learner().set_params(**params)

    def fit(self, training_data, classes):
        """
        Train the learner with the given data and known classes

        :param training_data: A multidimensional numpy array of training data
        :param classes: A numpy array of known classes
        :return: nothing
        """
        if self.learner() is None:
            return None

        return self.learner().fit(training_data, classes)

    def predict(self, data):
        """
        Have the learner predict classes given a set of data

        :param data: A multidimensional dumpy array of test data
        :return: The predicted classes for the test data
        """
        if self.learner() is None:
            return None

        return self.learner().predict(data)

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            print(msg.format(*args))

    def write_visualization(self, path):
        """
        Write a visualization of the given learner to the given path (including file name but not extension)
        :return: self
        """
        pass
