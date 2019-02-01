import itertools
import logging
import glob
import re

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None, multiple_runs=True,
                        x_scale='linear', y_scale='linear',
                        x_label='Training examples (count)', y_label='Accuracy (0.0 - 1.0)'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : list, array
        The training sizes

    train_scores : list, array
        The training scores

    test_scores : list, array
        The testing sizes

    multiple_runs : boolean
        If True, assume the given train and test scores represent multiple runs of a given test (the default)

    x_scale: string
        The x scale to use (defaults to None)

    y_scale: string
        The y scale to use (defaults to None)

    x_label: string
        Label fo the x-axis

    y_label: string
        Label fo the y-axis
    """
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()

    train_points = train_scores
    test_points = test_scores

    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    if multiple_runs:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2)

    plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
             label="Training score")
    plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_model_complexity_curve(title, train_sizes, train_scores, test_scores, ylim=None, multiple_runs=True,
                                x_scale='linear', y_scale='linear',
                                x_label='Training examples (count)', y_label='Accuracy (0.0 - 1.0)',
                                x_ticks=None, x_tick_labels=None, chart_type='line'):
    """
    Generate a simple plot of the test and training model complexity curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : list, array
        The training sizes

    train_scores : list, array
        The training scores

    test_scores : list, array
        The testing sizes

    multiple_runs : boolean
        If True, assume the given train and test scores represent multiple runs of a given test (the default)

    x_scale: string
        The x scale to use (defaults to None)

    y_scale: string
        The y scale to use (defaults to None)

    x_label: string
        Label fo the x-axis

    y_label: string
        Label fo the y-axis
    """
    plt.close('all')
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()

    train_points = train_scores
    test_points = test_scores

    ax = plt.gca()
    if x_scale is not None or y_scale is not None:
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    # TODO: https://stackoverflow.com/questions/2715535/how-to-plot-non-numeric-data-in-matplotlib
    if x_ticks is not None and x_tick_labels is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    train_scores_mean = None
    train_scores_std = None
    test_scores_mean = None
    test_scores_std = None
    if multiple_runs:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

    if chart_type == 'line':
        if multiple_runs:
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="salmon")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2, color="skyblue")

        plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
                 label="Training score")
        plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
                 label="Cross-validation score")
    if chart_type == 'bar':

        # https://matplotlib.org/examples/api/barchart_demo.html

        ind = train_sizes
        if x_tick_labels is not None:
            ind = np.arange(len(x_tick_labels))
            ax.set_xticklabels(x_tick_labels)

        bar_width = 0.35
        ax.bar(ind, train_points, bar_width, yerr=train_scores_std, label="Training score")
        ax.bar(ind + bar_width, test_points, bar_width, yerr=test_scores_std,
                        label="Cross-validation score")

        ax.grid(which='both')
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.set_xticks(ind + bar_width / 2)
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels, rotation=45)

    plt.legend(loc="best")
    plt.tight_layout()

    return plt


def plot_model_timing(title, data_sizes, fit_scores, predict_scores, ylim=None):
    """
    Generate a simple plot of the given model timing data

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    data_sizes : list, array
        The data sizes

    fit_scores : list, array
        The fit/train times

    predict_scores : list, array
        The predict times

    """
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Data Size (% of total)")
    plt.ylabel("Time (s)")
    fit_scores_mean = np.mean(fit_scores, axis=1)
    fit_scores_std = np.std(fit_scores, axis=1)
    predict_scores_mean = np.mean(predict_scores, axis=1)
    predict_scores_std = np.std(predict_scores, axis=1)
    plt.grid()
    plt.tight_layout()

    plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                     fit_scores_mean + fit_scores_std, alpha=0.2)
    plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                     predict_scores_mean + predict_scores_std, alpha=0.2)
    plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Predict time")
    plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Fit time")

    plt.legend(loc="best")
    return plt


# Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm: The matrix from metrics.confusion_matrics
    :param classes: The classes for the dataset
    :param normalize: If true, normalize
    :param title: The title for the plot
    :param cmap: The color map to use

    :return: The confusion matrix plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info('Confusion matrix, without normalization')

    logger.info(cm)

    plt.close()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def read_and_plot_reg_table(reg_file, output_dir, clf_name, dataset_readable_name):
    param_regex = re.compile('param_')
    train_regex = re.compile('split[0-9]+_train')
    test_regex = re.compile('split[0-9]+_test')
    data = pd.read_csv(reg_file)

    param_columns = list(filter(param_regex.match, data.columns))
    train_columns = list(filter(train_regex.match, data.columns))
    test_columns = list(filter(test_regex.match, data.columns))

    logger.info(reg_file)

    param_values = list(map(np.unique, [data[x].dropna() for x in param_columns]))
    logger.info(param_values)

    # Add columns that are the param values as strings for layer querying
    for param in param_columns:
        data['{}_str'.format(param)] = data[param].apply(lambda x: str(x))  #.astype(str)

    best_run = data[data['rank_test_score'] == 1]

    for i, param in enumerate(param_columns):
        if param_values[i].shape[0] == 1:
            continue

        logger.info("{} - {}".format(i, param, param_values[i]))

        other_params = ['{}_str'.format(x) for x in list(param_columns) if x != param]
        best_other_params = best_run[other_params]
        best_other_params = dict(zip(list(best_other_params.columns), list(best_other_params.iloc[0])))
        best_other_params_str = ' and '.join(['%s == "%s"' % (key, value) for (key, value) in
                                              best_other_params.items()])
        logger.info(other_params)
        logger.info(best_other_params)
        logger.info(best_other_params_str)

        values = param_values[i]
        param_data = data.query(best_other_params_str)
        if values.dtype != 'O':
            param_data = param_data.sort_values(by=[param])

        logger.info(param_data)
        logger.info(param_data.shape)
        if param_data.shape[0] == 0:
            continue

        x_tick_labels = None
        chart_type = 'line'
        if values.dtype == 'O':
            values = param_data.index
            x_tick_labels = param_values[i]
            chart_type = 'bar'

        x_scale = 'linear'
        if np.unique(np.ediff1d(values)).shape[0] > 1 and 'tol' not in param:
            x_scale = 'log'

        logger.info(values)
        logger.info(param_data[train_columns].values)
        logger.info(param_data[test_columns].values)

        param_name = param.split('__')[-1]
        mpl.rcParams.update(mpl.rcParamsDefault)
        plot = plot_model_complexity_curve(
            'Model Complexity: {} - {} ({})'.format(clf_name, dataset_readable_name, param_name),
            values,
            np.array(param_data[train_columns].values),
            np.array(param_data[test_columns].values),
            x_scale=x_scale,
            x_label=' '.join(map(lambda x: x.capitalize(), param_name.split('_'))),
            x_ticks=values,
            x_tick_labels=x_tick_labels,
            chart_type=chart_type,
            multiple_runs=True
        )
        plot.savefig('{}/images/{}_{}_{}_MC.png'.format(output_dir, clf_name, dataset_readable_name, param_name),
                     format='png', dpi=150)
        logger.info("----------")


if __name__ == '__main__':
    # read_and_plot_reg_table('output.final/ANN_HTRU2_reg.csv', 'output', 'ANN', 'HTRU2')
    reg_name_regex = re.compile('/([A-Za-z]+)_(.*)_reg.csv')

    reg_files = glob.glob('output.final/*_reg.csv')
    for reg_file in reg_files:
        clf_name, ds_name = reg_name_regex.search(reg_file).groups()
        read_and_plot_reg_table(reg_file, 'output', clf_name, ds_name)
