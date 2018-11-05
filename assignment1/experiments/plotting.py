import itertools

import numpy as np

import matplotlib.pyplot as plt


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
                                x_label='Training examples (count)', y_label='Accuracy (0.0 - 1.0)'):
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
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt
