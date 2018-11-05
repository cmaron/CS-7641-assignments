import os
from collections import defaultdict
from time import clock

import pandas as pd
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight

from .base import *
from .ANN import *
from .Boosting import *
from .DT import *
from .KNN import *
from .SVM import *
from .plotting import *

__all__ = ['ANN', 'Boosting', 'DT', 'KNN', 'SVM']

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


scorer = make_scorer(balanced_accuracy)


def basic_results(clf, classes, training_x, training_y, test_x, test_y, params, clf_type=None, dataset=None,
                  dataset_readable_name=None, seed=55, threads=1):
    print("Computing basic results ({} thread(s))".format(threads))

    if clf_type is None or dataset is None:
        raise Exception('clf_type and dataset are required')
    if seed is not None:
        np.random.seed(seed)

    cv = ms.GridSearchCV(clf, n_jobs=threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=scorer)
    cv.fit(training_x, training_y)
    reg_table = pd.DataFrame(cv.cv_results_)
    reg_table.to_csv('{}/{}_{}_reg.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
    test_score = cv.score(test_x, test_y)

    # TODO: Ensure this is an estimator that can handle this?
    best_estimator = cv.best_estimator_.fit(training_x, training_y)
    final_estimator = best_estimator._final_estimator
    best_params = pd.DataFrame([final_estimator.get_params()])
    best_params.to_csv('{}/{}_{}_best_params.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
    print(" - Grid search complete")

    final_estimator.write_visualization('{}/images/{}_{}_LC'.format(OUTPUT_DIRECTORY, clf_type, dataset))

    test_y_predicted = cv.predict(test_x)
    cnf_matrix = confusion_matrix(test_y, test_y_predicted)
    np.set_printoptions(precision=2)
    plt = plot_confusion_matrix(cnf_matrix, classes,
                                title='Confusion Matrix: {} - {}'.format(clf_type, dataset_readable_name))
    plt.savefig('{}/images/{}_{}_CM.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150,
                bbox_inches='tight')

    plt = plot_confusion_matrix(cnf_matrix, classes, normalize=True,
                                title='Normalized Confusion Matrix: {} - {}'.format(clf_type, dataset_readable_name))
    plt.savefig('{}/images/{}_{}_NCM.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150,
                bbox_inches='tight')

    print(" - Visualization complete")

    with open('{}/test results.csv'.format(OUTPUT_DIRECTORY), 'a') as f:
        f.write('{},{},{},{}\n'.format(clf_type, dataset, test_score, cv.best_params_))
    n = training_y.shape[0]
    # TODO: Is the range here dependent on the dataset?
    train_sizes = list(map(int, list(np.geomspace(max(n * 0.05, 50), n * 0.79, num=20, endpoint=True))))
    train_sizes, train_scores, test_scores = ms.learning_curve(
        cv.best_estimator_,
        training_x,
        training_y,
        cv=5,
        train_sizes=train_sizes,
        verbose=10,
        scoring=scorer,
        n_jobs=threads)
    curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)

    curve_train_scores.to_csv('{}/{}_{}_LC_train.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset))
    curve_test_scores.to_csv('{}/{}_{}_LC_test.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset))
    plt = plot_learning_curve('Learning Curve: {} - {}'.format(clf_type, dataset_readable_name),
                              train_sizes,
                              train_scores, test_scores)
    plt.savefig('{}/images/{}_{}_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)
    print(" - Learning curve complete")

    return cv


def iteration_lc(clf, training_x, training_y, test_x, test_y, params, clf_type=None, dataset=None,
                 dataset_readable_name=None, seed=55, threads=1):
    print("Building iteration learning curve for params {} ({} threads)".format(params, threads))

    if clf_type is None or dataset is None:
        raise Exception('clf_type and dataset are required')
    if seed is not None:
        np.random.seed(seed)

    cv = ms.GridSearchCV(clf, n_jobs=threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=scorer)
    cv.fit(training_x, training_y)
    reg_table = pd.DataFrame(cv.cv_results_)
    reg_table.to_csv('{}/ITER_base_{}_{}.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:
        d['param_{}'.format(name)].append(value)
        clf.set_params(**{name: value})
        clf.fit(training_x, training_y)
        pred = clf.predict(training_x)
        d['train acc'].append(balanced_accuracy(training_y, pred))
        clf.fit(training_x, training_y)
        pred = clf.predict(test_x)
        d['test acc'].append(balanced_accuracy(test_y, pred))
        print(' - {}'.format(value))
    d = pd.DataFrame(d)
    d.to_csv('{}/ITERtestSET_{}_{}.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
    plt = plot_learning_curve('{} - {} ({})'.format(clf_type, dataset_readable_name, name),
                              d['param_{}'.format(name)], d['train acc'], d['test acc'],
                              multiple_runs=False,  # x_scale='log',
                              x_label='Value')
    plt.savefig('{}/images/{}_{}_ITER_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)

    print(" - Iteration learning curve complete")

    return cv


def add_noise(y, frac=0.1):
    np.random.seed(456)
    n = y.shape[0]
    sz = int(n * frac)
    ind = np.random.choice(np.arange(n), size=sz, replace=False)
    tmp = y.copy()
    tmp[ind] = 1 - tmp[ind]
    return tmp


def make_timing_curve(x, y, clf, clf_name, dataset, dataset_readable_name, verbose=False, seed=42):
    print("Building timing curve")

    # np.linspace(0.1, 1, num=10)  #
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 5
    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, frac in enumerate(sizes):
        for j in range(tests):
            np.random.seed(seed)
            # TODO: Should this random_state be different than seed?
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=1 - frac, random_state=seed)
            st = clock()
            clf.fit(x_train, y_train)
            out['train'][i, j] = (clock() - st)
            st = clock()
            clf.predict(x_test)
            out['test'][i, j] = (clock() - st)
            if verbose:
                print(clf_name, dataset, frac)

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    plt = plot_model_timing('{} - {}'.format(clf_name, dataset_readable_name),
                            np.array(sizes) * 100, train_df, test_df)
    plt.savefig('{}/images/{}_{}_TC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset), format='png', dpi=150)

    out = pd.DataFrame(index=sizes)
    out['train'] = np.mean(train_df, axis=1)
    out['test'] = np.mean(test_df, axis=1)
    out.to_csv('{}/{}_{}_timing.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset))


def make_complexity_curve(x, y, param_name, param_display_name, param_values, clf, clf_name, dataset,
                          dataset_readable_name, x_scale, verbose=False, threads=1):
    print("Building model complexity curve")
    train_scores, test_scores = validation_curve(clf, x, y, param_name, param_values, cv=5, verbose=verbose,
                                                 scoring=scorer, n_jobs=threads)

    curve_train_scores = pd.DataFrame(index=param_values, data=train_scores)
    curve_test_scores = pd.DataFrame(index=param_values, data=test_scores)
    curve_train_scores.to_csv('{}/{}_{}_{}_MC_train.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name))
    curve_test_scores.to_csv('{}/{}_{}_{}_MC_test.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name))
    plt = plot_model_complexity_curve(
        'Model Complexity: {} - {} ({})'.format(clf_name, dataset_readable_name, param_display_name),
        param_values,
        train_scores, test_scores, x_scale=x_scale,
        x_label=param_display_name)
    plt.savefig('{}/images/{}_{}_{}_MC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name), format='png',
                dpi=150)

    print(" - Model complexity curve complete")


def perform_experiment(ds, ds_name, ds_readable_name, clf, clf_name, clf_label, params, timing_params=None,
                       iteration_pipe_params=None, iteration_params=None, complexity_param=None, seed=0, threads=1,
                       iteration_lc_only=False, verbose=False):
    # TODO: Fix this by changing the datatypes of the columns to float64?
    warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)
    warnings.simplefilter("ignore", DeprecationWarning)

    if verbose:
        print("Experimenting on {} with classifier {}.".format(ds_name, clf))
    ds_training_x, ds_testing_x, ds_training_y, ds_testing_y = ms.train_test_split(
        ds.features,
        ds.classes,
        test_size=0.2,
        random_state=seed,
        stratify=ds.classes)

    # Adjust training data if need be
    ds_training_x, ds_training_y = ds.pre_training_adjustment(ds_training_x, ds_training_y)

    # MinMaxScaler !?
    pipe = Pipeline([('Scale', StandardScaler()),
                     ('Cull1', SelectFromModel(RandomForestClassifier(random_state=seed), threshold='median')),
                     (clf_label, clf)])
    ds_final_params = None
    if not iteration_lc_only:
        ds_clf = basic_results(pipe, np.unique(ds.classes), ds_training_x, ds_training_y, ds_testing_x, ds_testing_y,
                               params, clf_name, ds_name, ds_readable_name, threads=threads, seed=seed)

        ds_final_params = ds_clf.best_params_
        if verbose:
            print("ds_final_params: {}".format(ds_final_params))

        pipe.set_params(**ds_final_params)

        if complexity_param is not None:
            param_display_name = complexity_param['name']
            x_scale = 'linear'
            if 'display_name' in complexity_param:
                param_display_name = complexity_param['display_name']
            if 'x_scale' in complexity_param:
                x_scale = complexity_param['x_scale']
            make_complexity_curve(ds.features, ds.classes, complexity_param['name'], param_display_name,
                                  complexity_param['values'], pipe,
                                  clf_name, ds_name, ds_readable_name, x_scale,
                                  threads=threads, verbose=verbose)

        if timing_params is not None:
            pipe.set_params(**timing_params)
        make_timing_curve(ds.features, ds.classes, pipe, clf_name, ds_name, ds_readable_name, seed=seed,
                          verbose=verbose)

    if iteration_params is not None:
        if iteration_pipe_params is not None:
            pipe.set_params(**iteration_pipe_params)
        iteration_lc(pipe, ds_training_x, ds_training_y, ds_testing_x, ds_testing_y, iteration_params,
                     clf_name, ds_name, ds_readable_name, threads=threads, seed=seed)

    # Return the best params found, if we have any
    return ds_final_params
