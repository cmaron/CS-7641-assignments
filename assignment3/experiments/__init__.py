import datetime
import warnings

from tempfile import mkdtemp

import sklearn
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve

from .base import *
from .benchmark import *
from .clustering import *
from .ICA import *
from .PCA import *
from .LDA import *
from .SVD import *
from .RF import *
from .RP import *
from .plotting import *
from .scoring import *

__all__ = ['pipeline_memory', 'run_subexperiment', 'clustering', 'benchmark', 'ICA', 'PCA', 'LDA', 'SVD', 'RF', 'RP']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# TODO: Fix this by changing the datatypes of the columns to float64?
warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)

warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# Keep a cache for the pipelines to speed things up
pipeline_cachedir = mkdtemp()
# pipeline_memory = Memory(cachedir=pipeline_cachedir, verbose=10)
pipeline_memory = None

# The best ANN params from assignment 1 (for just one dataset)
BEST_NN_PARAMS = {'NN__activation': ['relu'], 'NN__alpha': [1.0],
                  'NN__hidden_layer_sizes': [(36, 36)], 'NN__learning_rate_init': [0.016]}


def run_subexperiment(main_experiment, out, ds=None):
    if not os.path.exists(out):
        os.makedirs(out)

    out = out + '/{}'
    details = main_experiment.get_details()
    # Run the clustering again as a sub-experiment for this one
    clustering_details = ExperimentDetails(
        details.ds if not ds else ds,
        details.ds_name,
        details.ds_readable_name,
        details.threads,
        details.seed)
    ce = clustering.ClusteringExperiment(clustering_details, verbose=main_experiment.get_vebose())

    return ce.perform_for_subexperiment(out, main_experiment)


def basic_results(clf, classes, training_x, training_y, test_x, test_y, params, clf_type=None, dataset=None,
                  dataset_readable_name=None, binary_classification=False, best_params=None, seed=55, threads=1):
    logger.info("Computing basic results for {} ({} thread(s))".format(clf_type, threads))

    if clf_type is None or dataset is None:
        raise Exception('clf_type and dataset are required')
    if seed is not None:
        np.random.seed(seed)

    curr_scorer = scorer
    if binary_classification:
        curr_scorer = f1_scorer

    if best_params:
        clf.fit(training_x, training_y)
        test_score = clf.score(test_x, test_y)
        cv = clf
    else:
        cv = ms.GridSearchCV(clf, n_jobs=threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=curr_scorer)
        cv.fit(training_x, training_y)
        reg_table = pd.DataFrame(cv.cv_results_)
        reg_table.to_csv('{}/{}_{}_reg.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
        test_score = cv.score(test_x, test_y)

        # TODO: Ensure this is an estimator that can handle this?
        best_estimator = cv.best_estimator_.fit(training_x, training_y)
        final_estimator = best_estimator._final_estimator
        best_params = pd.DataFrame([final_estimator.get_params()])
        best_params.to_csv('{}/{}_{}_best_params.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
        logger.info(" - Grid search complete")

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

        logger.info(" - Visualization complete")

        with open('{}/test results.csv'.format(OUTPUT_DIRECTORY), 'a') as f:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%s')
            f.write('"{}",{},{},{},"{}"\n'.format(ts, clf_type, dataset, test_score, cv.best_params_))

    n = training_y.shape[0]

    train_sizes = np.append(np.linspace(0.05, 0.1, 20, endpoint=False),
                            np.linspace(0.1, 1, 20, endpoint=True))
    logger.info(" - n: {}, train_sizes: {}".format(n, train_sizes))
    train_sizes, train_scores, test_scores = ms.learning_curve(
        clf if best_params is not None else cv.best_estimator_,
        training_x,
        training_y,
        cv=5,
        train_sizes=train_sizes,
        verbose=10,
        scoring=curr_scorer,
        n_jobs=threads,
        random_state=seed)
    logger.info(" - n: {}, train_sizes: {}".format(n, train_sizes))
    curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)

    curve_train_scores.to_csv('{}/{}_{}_LC_train.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset))
    curve_test_scores.to_csv('{}/{}_{}_LC_test.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset))
    plt = plot_learning_curve('Learning Curve: {} - {}'.format(clf_type, dataset_readable_name),
                              train_sizes,
                              train_scores, test_scores)
    plt.savefig('{}/images/{}_{}_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)
    logger.info(" - Learning curve complete")

    return cv


def iteration_lc(clf, training_x, training_y, test_x, test_y, params, clf_type=None, dataset=None,
                 dataset_readable_name=None, binary_classification=False, x_scale='linear', seed=55, threads=1):
    logger.info("Building iteration learning curve for params {} ({} threads)".format(params, threads))

    if clf_type is None or dataset is None:
        raise Exception('clf_type and dataset are required')
    if seed is not None:
        np.random.seed(seed)

    curr_scorer = scorer
    acc_method = balanced_accuracy
    if binary_classification:
        curr_scorer = f1_scorer
        acc_method = f1_accuracy

    cv = ms.GridSearchCV(clf, n_jobs=threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=curr_scorer)
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
        d['train acc'].append(acc_method(training_y, pred))
        clf.fit(training_x, training_y)
        pred = clf.predict(test_x)
        d['test acc'].append(acc_method(test_y, pred))
        logger.info(' - {}'.format(value))
    d = pd.DataFrame(d)
    d.to_csv('{}/ITERtestSET_{}_{}.csv'.format(OUTPUT_DIRECTORY, clf_type, dataset), index=False)
    plt = plot_learning_curve('{} - {} ({})'.format(clf_type, dataset_readable_name, name),
                              d['param_{}'.format(name)], d['train acc'], d['test acc'],
                              multiple_runs=False, x_scale=x_scale,
                              x_label='Value')
    plt.savefig('{}/images/{}_{}_ITER_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)

    logger.info(" - Iteration learning curve complete")

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
    logger.info("Building timing curve")

    # np.linspace(0.1, 1, num=10)  #
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 5
    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, frac in enumerate(sizes):
        for j in range(tests):
            np.random.seed(seed)
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=1 - frac, random_state=seed)
            st = clock()
            clf.fit(x_train, y_train)
            out['train'][i, j] = (clock() - st)
            st = clock()
            clf.predict(x_test)
            out['test'][i, j] = (clock() - st)
            logger.info(" - {} {} {}".format(clf_name, dataset, frac))

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    plt = plot_model_timing('{} - {}'.format(clf_name, dataset_readable_name),
                            np.array(sizes) * 100, train_df, test_df)
    plt.savefig('{}/images/{}_{}_TC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset), format='png', dpi=150)

    out = pd.DataFrame(index=sizes)
    out['train'] = np.mean(train_df, axis=1)
    out['test'] = np.mean(test_df, axis=1)
    out.to_csv('{}/{}_{}_timing.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset))

    logger.info(" - Timing curve complete")


def make_complexity_curve(x, y, param_name, param_display_name, param_values, clf, clf_name, dataset,
                          dataset_readable_name, x_scale, verbose=False, binary_classification=False, threads=1):
    logger.info("Building model complexity curve")
    curr_scorer = scorer
    if binary_classification:
        curr_scorer = f1_scorer

    train_scores, test_scores = validation_curve(clf, x, y, param_name, param_values, cv=5, verbose=verbose,
                                                 scoring=curr_scorer, n_jobs=threads)

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

    logger.info(" - Model complexity curve complete")


def perform_experiment(ds, ds_name, ds_readable_name, clf, clf_name, clf_label, params, timing_params=None,
                       iteration_details=None, complexity_param=None, seed=0, threads=1,
                       iteration_lc_only=False, best_params=None, verbose=False):
    # TODO: Fix this by changing the datatypes of the columns to float64?
    warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)
    warnings.simplefilter("ignore", DeprecationWarning)

    logger.info("Experimenting on {} with classifier {}.".format(ds_name, clf))

    ds_training_x, ds_testing_x, ds_training_y, ds_testing_y = ms.train_test_split(
        ds.features,
        ds.classes,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=ds.classes)

    # Adjust training data if need be
    ds_training_x, ds_training_y = ds.pre_training_adjustment(ds_training_x, ds_training_y)

    pipe = Pipeline([('Scale', StandardScaler()),
                     (clf_label, clf)])
    ds_final_params = None
    if not iteration_lc_only:
        ds_clf = basic_results(pipe, np.unique(ds.classes), ds_training_x, ds_training_y, ds_testing_x, ds_testing_y,
                               params, clf_name, ds_name, ds_readable_name, binary_classification=ds.binary,
                               best_params=best_params, threads=threads, seed=seed)

        if best_params is not None:
            ds_final_params = best_params
        else:
            ds_final_params = ds_clf.best_params_
            pipe.set_params(**ds_final_params)

        if verbose:
            logger.info("ds_final_params: {}".format(ds_final_params))

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
                                  binary_classification=ds.binary,
                                  threads=threads, verbose=verbose)

        if timing_params is not None:
            pipe.set_params(**timing_params)
        make_timing_curve(ds.features, ds.classes, pipe, clf_name, ds_name, ds_readable_name,
                          seed=seed, verbose=verbose)

    if iteration_details is not None:
        x_scale = 'linear'
        if 'pipe_params' in iteration_details:
            pipe.set_params(**iteration_details['pipe_params'])
        if 'x_scale' in iteration_details:
            x_scale = iteration_details['x_scale']

        iteration_lc(pipe, ds_training_x, ds_training_y, ds_testing_x, ds_testing_y, iteration_details['params'],
                     clf_name, ds_name, ds_readable_name, x_scale=x_scale,
                     binary_classification=ds.binary, threads=threads, seed=seed)

    # Return the best params found, if we have any
    return ds_final_params
