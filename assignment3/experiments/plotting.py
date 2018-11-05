import itertools
import logging
import os
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from kneed import KneeLocator

from matplotlib import cycler
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from os.path import basename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# input_path = 'output.final/'
# output_path = 'output.final/images/'
input_path = 'output/'
output_path = 'output/images/'
to_process = {
    'benchmark': {
        'path': 'benchmark',
        'nn_curve': False,
        'multiple_trials': False
    },
    'ICA': {
        'path': 'ICA',
        'nn_curve': False,
        'multiple_trials': False
    },
    'PCA': {
        'path': 'PCA',
        'nn_curve': False,
        'multiple_trials': False
    },
    'RF': {
        'path': 'RF',
        'nn_curve': False,
        'multiple_trials': False
    },
    'SVD': {
        'path': 'SVD',
        'nn_curve': False,
        'multiple_trials': False
    },
    'RP': {
        'path': 'RP',
        'nn_curve': False,
        'multiple_trials': False
    }
}

the_best = {}

# File name regex to pull dataset name
scree_file_name_regex = re.compile('(.*)_scree\.csv')
multi_scree_file_name_regex = re.compile('(.*)_scree(.*)\.csv')
sse_file_name_regex = re.compile('(.*)_sse\.csv')
acc_file_name_regex = re.compile('(.*)_acc\.csv')
adj_mi_file_name_regex = re.compile('(.*)_adj_mi\.csv')
loglikelihood_file_name_regex = re.compile('(.*)_logliklihood\.csv')
bic_file_name_regex = re.compile('(.*)_bic\.csv')
tsne_file_name_regex = re.compile('(.*)_2D\.csv')
sil_score_file_name_regex = re.compile('(.*)_sil_score\.csv')
sil_samples_file_name_regex = re.compile('(.*)_sil_samples\.csv')

algos = {
    'scree': {
        'regex': scree_file_name_regex,
        'descriptive_name': 'Scree'
    },
    'sse': {
        'regex': sse_file_name_regex,
        'descriptive_name': 'SSE'
    },
    'acc': {
        'regex': acc_file_name_regex,
        'descriptive_name': 'Accuracy'
    },
    'adj_mi': {
        'regex': adj_mi_file_name_regex,
        'descriptive_name': 'Adjusted Mutual Information'
    },
    'loglikelihood': {
        'regex': loglikelihood_file_name_regex,
        'descriptive_name': 'Log Likelihood'
    },
    'BIC': {
        'regex': bic_file_name_regex,
        'descriptive_name': 'BIC'
    },
    'sil_score': {
        'regex': sil_score_file_name_regex,
        'descriptive_name': 'Silhouette Score'
    },
    'tsne': {
        'regex': tsne_file_name_regex,
        'descriptive_name': 't-SNE'
    }
}

WATERMARK = False
GATECH_USERNAME = 'DO NOT STEAL'
TERM = 'Fall 2018'


def watermark(p):
    if not WATERMARK:
        return p

    ax = plt.gca()
    for i in range(1, 11):
        p.text(0.95, 0.95 - (i * (1.0/10)), '{} {}'.format(GATECH_USERNAME, TERM), transform=ax.transAxes,
               fontsize=32, color='gray',
               ha='right', va='bottom', alpha=0.2)
    return p


# Adapted from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
def find_knee(values):
    # get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    # np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def plot_scree(title, df, problem_name, multiple_runs=False, xlabel='Number of Clusters', ylabel=None):
    if ylabel is None:
        ylabel = 'Kurtosis'
        if problem_name == 'PCA' or problem_name == 'SVD':
            ylabel = 'Variance'
        elif problem_name == 'RP':
            # ylabel = 'PDCC'  # 'Pairwise distance corrcoef'
            ylabel = 'Pairwise distance corrcoef'
        elif problem_name == 'RF':
            ylabel = 'Feature Importances'
    title = title.format(ylabel)

    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    ax = plt.gca()

    x_points = df.index.values
    y_points = df[1]
    if multiple_runs:
        y_points = np.mean(df.iloc[:, 1:-1], axis=1)
        y_std = np.std(df.iloc[:, 1:-1], axis=1)
        plt.plot(x_points, y_points, 'o-', linewidth=1, markersize=2,
                 label=ylabel)
        plt.fill_between(x_points, y_points - y_std,
                         y_points + y_std, alpha=0.2)
    else:
        plt.plot(x_points, y_points, 'o-', linewidth=1, markersize=2,
                 label=ylabel)

    min_value = np.min(y_points)
    min_point = y_points.idxmin()
    max_value = np.max(y_points)
    max_point = y_points.idxmax()
    knee_point = find_knee(y_points)
    kl = KneeLocator(x_points, y_points)

    ax.axvline(x=min_point, linestyle="--", label="Min: {}".format(int(min_point)))
    ax.axvline(x=max_point, linestyle="--", label="Max: {}".format(int(max_point)))
    if kl.knee_x is not None:
        ax.axvline(x=kl.knee_x, linestyle="--", label="Knee: {}".format(kl.knee_x))
    else:
        ax.axvline(x=knee_point, linestyle="--", label="Knee: {}".format(knee_point))

    ax.set_xticks(df.index.values, minor=False)

    plt.legend(loc="best")

    return plt


def plot_kmeans_gmm(title, df, xlabel='Number of Clusters', ylabel='Accuracy'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df['Kmeans'], 'o-', linewidth=1, markersize=2,
             label="k-Means")
    plt.plot(df.index.values, df['GMM'], 'o-', linewidth=1, markersize=2,
             label="GMM")
    plt.legend(loc="best")

    return plt


def plot_sse(title, df, xlabel='Number of Clusters', ylabel='SSE'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


def plot_loglikelihood(title, df, xlabel='Number of Clusters', ylabel='Log Likelihood'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


def plot_bic(title, df, xlabel='Number of Clusters', ylabel='BIC'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


INITIAL_FIG_SIZE = plt.rcParams["figure.figsize"]


def plot_combined(title, df, data_columns, tsne_data=None, extra_data=None, extra_data_name=None,
                  xlabel='Number of Clusters', ylabel=None):
    plt.close()
    plt.figure()

    if tsne_data is not None:
        plt.rcParams["figure.figsize"] = (INITIAL_FIG_SIZE[0] * 1.25,
                                          INITIAL_FIG_SIZE[1])
        f, (ax1, ax2) = plt.subplots(1, 2)
    else:
        f, (ax1) = plt.subplots(1, 1)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    if ylabel:
        ax1.set_ylabel(ylabel)
    color_len = len(data_columns)
    if extra_data is not None:
        color_len += 2

    for column in data_columns:
        if column == 'tsne':
            pass
        else:
            ax1.plot(df.index.values, df[column], linewidth=1,
                     label=algos[column]['descriptive_name'])
    if tsne_data is not None:
        ax2.scatter(tsne_data['x'], tsne_data['y'], c=tsne_data['target'], alpha=0.7, s=5)
        ax2.xaxis.set_major_formatter(NullFormatter())
        ax2.yaxis.set_major_formatter(NullFormatter())
        ax2.grid(None)
        ax2.set_xticks([])
        ax2.set_yticks([])

    if extra_data is not None and extra_data_name is not None:
        ex_ax = ax1.twinx()
        ex_ax.plot(extra_data.index.values, extra_data.iloc[:, 0], linewidth=1,
                   label=extra_data_name)
        ex_ax.set_ylabel(extra_data_name)
        ex_ax.tick_params('y')

    ax1.legend(loc="best")
    ax1.grid()
    ax1.axis('tight')

    f.tight_layout()

    return plt


def plot_tsne(title, df):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    possible_clusters = list(set(df['target']))

    ax = plt.gca()
    ax.set_title(title)
    ax.scatter(df['x'], df['y'], c=df['target'], alpha=0.7, s=5)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.grid(None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')

    return plt


def plot_adj_mi(title, df):
    return plot_kmeans_gmm(title, df, ylabel='Adj. MI')


def plot_acc(title, df):
    return plot_kmeans_gmm(title, df)


def plot_sil_score(title, df):
    return plot_kmeans_gmm(title, df, 'Number of Clusters', 'Silhouette Score')


# Adapted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def plot_sil_samples(title, df, n_clusters):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    df = df[df['k'] == n_clusters]
    ax = plt.gca()
    # Create a subplot with 1 row and 2 columns
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(18, 7)
    sample_silhouette_values = df[df['type'] == 'Kmeans']['score'].astype(np.double)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    x_min = float(min(sample_silhouette_values))
    x_max = float(max(sample_silhouette_values))
    ax.set_xlim([x_min - 0.05, x_max + 0.05])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, df.shape[0]/2 + (n_clusters + 1) * 10])

    cluster_labels = df[df['type'] == 'Kmeans']['label'].astype(np.float).values
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,4))
    # ax.xaxis.major.formatter._useMathText = True
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i].values
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        ith_cluster_silhouette_values.sort()

        y_upper = y_lower + size_cluster_i

        # color = colors[i]  # cm.nipy_spectral(float(i) / n_clusters)
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         x_min, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(x_min-0.02, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # TODO: Get this to work?
    # # The vertical line for average silhouette score of all the values
    # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(np.linspace(round(x_min, 2), round(x_max, 2), 7))
    # ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # TODO: Get this to work?
    # # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')

    # TODO: Get this to work?
    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')
    #
    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    # plt.suptitle(("Silhouette analysis for KMeans clustering on data "
    #               "with n_clusters = %d" % n_clusters),
    #              fontsize=14, fontweight='bold')

    return plt


def get_ds_readable_name(ds_name):
    return ' '.join(map(lambda x: x.capitalize(), ds_name.split('_')))


def get_ds_name(file, regexp):
    search_result = regexp.search(basename(file))
    if search_result is None:
        return False, False

    ds_name = search_result.groups()[0]

    return ds_name, get_ds_readable_name(ds_name)


def read_and_plot_scree(problem, file, output_dir):
    multi_scree = False
    scree_index = False

    ds_name, ds_readable_name = get_ds_name(file, scree_file_name_regex)
    if not ds_name and not ds_readable_name:
        ds_name, ds_readable_name = get_ds_name(file, multi_scree_file_name_regex)
        scree_index = multi_scree_file_name_regex.search(basename(file)).groups()[1]
        multi_scree = True

    ylabel = None
    if multi_scree and scree_index == '2':
        ylabel = 'Reconstruction Error'

    logger.info("Plotting scree for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: '.format(ds_readable_name, problem['name']) + '{} vs Number of Components'
    df = pd.read_csv(file, header=None).dropna().set_index(0)
    p = plot_scree(title, df, problem['name'], multiple_runs=multi_scree, xlabel='Number of Components',
                   ylabel=ylabel)
    p = watermark(p)
    if multi_scree:
        logger.info('{}/{}/{}_scree_{}.png {}'.format(output_dir, problem['name'], ds_name, scree_index, ylabel))
        p.savefig(
            '{}/{}/{}_scree_{}.png'.format(output_dir, problem['name'], ds_name, scree_index),
            format='png', bbox_inches='tight', dpi=150)
    else:
        p.savefig(
            '{}/{}/{}_scree.png'.format(output_dir, problem['name'], ds_name),
            format='png', bbox_inches='tight', dpi=150)


def read_and_plot_tsne(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, tsne_file_name_regex)
    logger.info("Plotting t-SNE for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file)
    p = plot_tsne(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_tsne.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_sse(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, sse_file_name_regex)
    logger.info("Plotting SSE for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: SSE vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_sse(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_sse.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_acc(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, acc_file_name_regex)
    logger.info("Plotting ACC for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: Accuracy vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_sse(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_acc.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_adj_mi(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, adj_mi_file_name_regex)
    logger.info("Plotting adj MI for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: Adj. MI vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_adj_mi(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_adj_mi.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_loglikelihood(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, loglikelihood_file_name_regex)
    logger.info("Plotting Log Likelihood for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: Log Likelihood vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_loglikelihood(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_loglikelihood.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_bic(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, bic_file_name_regex)
    logger.info("Plotting BIC for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: BIC vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_bic(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_bic.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_sil_score(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, sil_score_file_name_regex)
    logger.info("Plotting silhouette scores for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: Silhouette Score vs Number of Clusters'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file).set_index('k')
    p = plot_sil_score(title, df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_sil_score.png'.format(output_dir, problem['name'], ds_name),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_sil_samples(problem, file, output_dir):
    ds_name, ds_readable_name = get_ds_name(file, sil_samples_file_name_regex)
    logger.info("Plotting silhouette samples for file {} to {} ({})".format(file, output_dir, ds_name))

    title = '{} - {}: Silhouette Samples'.format(ds_readable_name, problem['name'])
    df = pd.read_csv(file)
    cluster_sizes = list(set(df['k']))
    for k in cluster_sizes:
        logger.info(" - Processing k={}".format(k))
        p = plot_sil_samples(title, df, k)
        p = watermark(p)
        p.savefig(
            '{}/{}/{}_sil_samples_{}.png'.format(output_dir, problem['name'], ds_name, k),
            format='png', bbox_inches='tight', dpi=150)


def read_and_plot_combined(problem, clustering_algo, ds_name, ds_readable_name, files, output_dir):
    logger.info("Plotting combined plot for files {} to {} ({})".format(files, output_dir, ds_name))
    title = '{} - {}: {}'.format(ds_readable_name, problem['name'], clustering_algo)

    plot_df = pd.DataFrame()
    tsne_df = pd.DataFrame()
    extra_df = pd.DataFrame()
    extra_name = None
    data_columns = sorted(files.keys())
    for c in data_columns:
        df = pd.read_csv(files[c])
        if clustering_algo == 'Kmeans' and c == 'sse':
            df = df.set_index('k')
            extra_df = df
            extra_name = 'SSE'
        elif clustering_algo == 'GMM' and c == 'BIC':
            df = df.set_index('k')
            extra_df = df
            extra_name = 'BIC'
        elif c == 'tsne':
            tsne_df = df
        elif c != 'sse' and c != 'BIC':
            df = df.set_index('k')
            plot_df[c] = df[clustering_algo]

    # Trim the extra columns
    data_columns = [k for k in data_columns if k != 'sse' and k != 'BIC']

    p = plot_combined(title, plot_df, data_columns, tsne_data=None, extra_data=extra_df, extra_data_name=extra_name,
                      xlabel='Number of Clusters', ylabel='Value')
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_{}_combined.png'.format(output_dir, problem['name'], ds_name, clustering_algo),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_problem(problem_name, problem, output_dir):
    problem['name'] = problem_name
    problem_path = '{}/{}'.format(input_path, problem['path'])

    if not os.path.exists('{}/{}'.format(output_dir, problem['name'])):
        os.makedirs('{}/{}'.format(output_dir, problem['name']))

    scree_files = glob.glob('{}/*_scree*.csv'.format(problem_path))
    logger.info("Scree files {}".format(scree_files))
    [read_and_plot_scree(problem, f, output_dir) for f in scree_files]

    clustering_tsne_files = glob.glob('{}/clustering/*_2D.csv'.format(problem_path))
    logger.info("Clustering t-SNE files {}".format(clustering_tsne_files))
    [read_and_plot_tsne(problem, f, output_dir) for f in clustering_tsne_files]

    clustering_sse_files = glob.glob('{}/clustering/*_sse.csv'.format(problem_path))
    logger.info("Clustering SSE files {}".format(clustering_sse_files))
    [read_and_plot_sse(problem, f, output_dir) for f in clustering_sse_files]

    clustering_acc_files = glob.glob('{}/clustering/*_acc.csv'.format(problem_path))
    logger.info("Clustering ACC files {}".format(clustering_acc_files))
    [read_and_plot_acc(problem, f, output_dir) for f in clustering_acc_files]

    clustering_adj_mi_files = glob.glob('{}/clustering/*_adj_mi.csv'.format(problem_path))
    logger.info("Clustering Adj MI files {}".format(clustering_adj_mi_files))
    [read_and_plot_adj_mi(problem, f, output_dir) for f in clustering_adj_mi_files]

    clustering_loglikelihood_files = glob.glob('{}/clustering/*_logliklihood.csv'.format(problem_path))
    logger.info("Clustering Log Liklihood files {}".format(clustering_loglikelihood_files))
    [read_and_plot_loglikelihood(problem, f, output_dir) for f in clustering_loglikelihood_files]

    clustering_bic_files = glob.glob('{}/clustering/*_bic.csv'.format(problem_path))
    logger.info("Clustering BIC files {}".format(clustering_bic_files))
    [read_and_plot_bic(problem, f, output_dir) for f in clustering_bic_files]

    clustering_sil_score = glob.glob('{}/clustering/*_sil_score.csv'.format(problem_path))
    logger.info("Clustering Sil score files {}".format(clustering_sil_score))
    [read_and_plot_sil_score(problem, f, output_dir) for f in clustering_sil_score]

    clustering_sil_sample_score = glob.glob('{}/clustering/*_sil_samples.csv'.format(problem_path))
    logger.info("Clustering Sil samples files {}".format(clustering_sil_sample_score))
    [read_and_plot_sil_samples(problem, f, output_dir) for f in clustering_sil_sample_score]

    combined_files = defaultdict(dict)
    combined_file_types = {'acc': clustering_acc_files, 'adj_mi': clustering_adj_mi_files,
                           'BIC': clustering_bic_files, 'sse': clustering_sse_files,
                           'sil_score': clustering_sil_score, 'tsne': clustering_tsne_files}
    for k in sorted(combined_file_types.keys()):
        for f in combined_file_types[k]:
            ds_name, ds_readable_name = get_ds_name(f, algos[k]['regex'])
            combined_files[ds_name][k] = f
    logger.info("Clustering combined files {}".format(combined_files))
    for k in sorted(combined_files.keys()):
        read_and_plot_combined(problem, 'Kmeans', k, get_ds_readable_name(k), combined_files[k], output_dir)
        read_and_plot_combined(problem, 'GMM', k, get_ds_readable_name(k), combined_files[k], output_dir)


def plot_results():
    for problem_name in to_process:
        logger.info("Processing {}".format(problem_name))
        problem = to_process[problem_name]

        read_and_plot_problem(problem_name, problem, output_path)


if __name__ == '__main__':
    plot_results()
