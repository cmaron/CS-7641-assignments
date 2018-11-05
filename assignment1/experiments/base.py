from abc import ABC, abstractmethod


class ExperimentDetails(object):
    def __init__(self, ds, ds_name, ds_readable_name, threads, seed):
        self.ds = ds
        self.ds_name = ds_name
        self.ds_readable_name = ds_readable_name
        self.threads = threads
        self.seed = seed


class BaseExperiment(ABC):
    def __init__(self, details):
        self._details = details

    @abstractmethod
    def perform(self):
        pass

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            print(msg.format(*args))
