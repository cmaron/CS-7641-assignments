from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.utils import compute_sample_weight


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


def f1_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return f1_score(truth, pred, average="binary", sample_weight=wts)


scorer = make_scorer(balanced_accuracy)
f1_scorer = make_scorer(f1_accuracy)


def get_scorer(dataset):
    if dataset.balanced:
        return f1_scorer, f1_accuracy

    return scorer, balanced_accuracy
