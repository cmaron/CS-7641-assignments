import glob
import json
import logging
import matplotlib.patheffects as path_effects
import numpy as np
import os
import pandas as pd
import re

from os.path import basename
from matplotlib import pyplot as plt
from shutil import copyfile


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# INPUT_PATH = 'output.final/'
# OUTPUT_PATH = 'output.final/images/'
# REPORT_PATH = 'output.final/report/'
INPUT_PATH = 'output/'
OUTPUT_PATH = 'output/images/'
REPORT_PATH = 'output/report/'

if not os.path.exists(REPORT_PATH):
    os.makedirs(REPORT_PATH)

TO_PROCESS = {
    'PI': {
        'path': 'PI',
        'file_regex': re.compile('(.*)_grid\.csv')
    },
    'VI': {
        'path': 'VI',
        'file_regex': re.compile('(.*)_grid\.csv')
    },
    'Q': {
        'path': 'Q',
        'file_regex': re.compile('(.*)_grid\.csv')
    }
}

the_best = {}

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


def plot_episode_stats(title_base, stats, smoothing_window=50):
    # Trim the DF down based on the episode lengths
    stats = stats[stats['length'] > 0]

    # Plot the episode length over time, both as a line and histogram
    fig1 = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    plt.plot(stats['length'])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.subplot(122)
    plt.hist(stats['length'], zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.title(title_base.format("Episode Length (Histogram)"))
    fig1 = watermark(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats['reward']).rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time ({})".format(smoothing_window))
    plt.subplot(122)
    plt.tight_layout()
    plt.hist(stats['reward'], zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title(title_base.format("Episode Reward (Histogram)"))
    fig2 = watermark(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    time_steps = np.cumsum(stats['time'])
    plt.plot(time_steps, np.arange(len(stats['time'])))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.subplot(122)
    plt.tight_layout()
    plt.hist(time_steps, zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.title(title_base.format("Episode Time (Histogram)"))
    fig3 = watermark(fig3)

    return fig1, fig2, fig3


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()

    return watermark(plt)


def plot_value_map(title, v, map_desc, color_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, v.shape[1]), ylim=(0, v.shape[0]))
    font_size = 'x-large'
    if v.shape[1] > 16:
        font_size = 'small'

    v_min = np.min(v)
    v_max = np.max(v)
    bins = np.linspace(v_min, v_max, 100)
    v_red = np.digitize(v, bins)/100.0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            value = np.round(v[i, j], 2)
            if len(str(value)) > 4:
                font_size = 'small'

    plt.title(title)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            y = v.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            value = np.round(v[i, j], 2)

            red = v_red[i, j]
            text2 = ax.text(x+0.5, y+0.5, value, size=font_size,
                            horizontalalignment='center', verticalalignment='center', color=(1.0, 1.0-red, 1.0-red))
            text2.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                   path_effects.Normal()])

    plt.axis('off')
    plt.xlim((0, v.shape[1]))
    plt.ylim((0, v.shape[0]))
    plt.tight_layout()

    return watermark(plt)


def plot_time_vs_steps(title, df, xlabel="Steps", ylabel="Time (s)"):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df['time'], '-', linewidth=1)
    plt.legend(loc="best")

    return watermark(plt)


def plot_reward_and_delta_vs_steps(title, df, xlabel="Steps", ylabel="Reward"):
    plt.close()
    plt.figure()

    f, (ax) = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    lns1 = ax.plot(df.index.values, df['reward'], linewidth=1, label=ylabel)

    ex_ax = ax.twinx()
    lns2 = ex_ax.plot(df.index.values, df['delta'], linewidth=1, label='Delta')
    ex_ax.set_ylabel('Delta')
    ex_ax.tick_params('y')

    ax.grid()
    ax.axis('tight')

    f.tight_layout()

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    return watermark(plt)


# Adapted from http://code.activestate.com/recipes/578293-unicode-command-line-histograms/
def cli_hist(data, bins=10):
    bars = u' ▁▂▃▄▅▆▇█'
    n, bin_edges = np.histogram(data, bins=bins)
    n2 = map(int, np.floor(n*(len(bars)-1)/(max(n))))
    res = u' '.join(bars[i] for i in n2)

    return res


# Adapted from https://gist.github.com/joezuntz/2f3bdc2ab0ea59229907
def ascii_hist(data, bins=10):
    N, X = np.histogram(data, bins=bins)
    total = 1.0 * len(data)
    width = 50
    nmax = N.max()
    lines = []

    for (xi, n) in zip(X, N):
        bar = '#' * int(n * 1.0 * width / nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        lines.append('{0}| {1}'.format(xi, bar))

    return lines


def fetch_mdp_name(file, regexp):
    search_result = regexp.search(basename(file))
    if search_result is None:
        return False, False

    mdp_name = search_result.groups()[0]

    return mdp_name, ' '.join(map(lambda x: x.capitalize(), mdp_name.split('_')))


def process_params(problem_name, params):
    param_str = '{}'.format(params['discount_factor'])
    if problem_name == 'Q':
        param_str = '{}_{}_{}_{}_{}'.format(params['alpha'], params['q_init'], params['epsilon'],
                                            params['epsilon_decay'], params['discount_factor'])

    return param_str


def find_optimal_params(problem_name, base_dir, file_regex):
    grid_files = glob.glob('{}/*_grid*.csv'.format(base_dir))
    logger.info("Grid files {}".format(grid_files))
    best_params = {}
    for f in grid_files:
        mdp, readable_mdp = fetch_mdp_name(f, file_regex)
        logger.info("MDP: {}, Readable MDP: {}".format(mdp, readable_mdp))
        df = pd.read_csv(f)
        best = df.copy()
        # Attempt to find the best params. First look at the reward mean, then median, then max. If at any point we
        # have more than one result as "best", try the next criterion
        for criterion in ['reward_mean', 'reward_median', 'reward_max']:
            best_value = np.max(best[criterion])
            best = best[best[criterion] == best_value]
            if best.shape[0] == 1:
                break

        # If we have more than one best, take the highest index.
        if best.shape[0] > 1:
            best = best.iloc[-1:]

        params = best.iloc[-1]['params']
        params = json.loads(params)
        best_index = best.iloc[-1].name

        best_params[mdp] = {
            'name': mdp,
            'readable_name': readable_mdp,
            'index': best_index,
            'params': params,
            'param_str': process_params(problem_name, params)
        }

    return best_params


def find_policy_images(base_dir, params):
    policy_images = {}
    for mdp in params:
        mdp_params = params[mdp]
        image_files = glob.glob('{}/{}_{}*.png'.format(base_dir, mdp_params['name'], mdp_params['param_str']))

        if len(image_files) == 2:
            policy_file = None
            value_file = None
            for image_file in image_files:
                if 'Value' in image_file:
                    value_file = image_file
                else:
                    policy_file = image_file

            logger.info("Value file {}, Policy File: {}".format(value_file, policy_file))
            policy_images[mdp] = {
                'value': value_file,
                'policy': policy_file
            }
        else:
            logger.error("Unable to find image file for {} with params {}".format(mdp, mdp_params))

    return policy_images


def find_data_files(base_dir, params):
    data_files = {}
    for mdp in params:
        mdp_params = params[mdp]
        files = glob.glob('{}/{}_{}.csv'.format(base_dir, mdp_params['name'], mdp_params['param_str']))
        optimal_files = glob.glob('{}/{}_{}_optimal.csv'.format(base_dir, mdp_params['name'], mdp_params['param_str']))
        episode_files = glob.glob('{}/{}_{}_episode.csv'.format(base_dir, mdp_params['name'], mdp_params['param_str']))
        logger.info("files {}".format(files))
        logger.info("optimal_files {}".format(optimal_files))
        logger.info("episode_files {}".format(episode_files))
        data_files[mdp] = {
            'file': files[0],
            'optimal_file': optimal_files[0]
        }
        if len(episode_files) > 0:
            data_files[mdp]['episode_file'] = episode_files[0]

    return data_files


def copy_best_images(best_images, base_dir):
    for problem_name in best_images:
        for mdp in best_images[problem_name]:
            mdp_files = best_images[problem_name][mdp]

            dest_dir = base_dir + '/' + problem_name
            policy_image = mdp_files['policy']
            value_image = mdp_files['value']

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            policy_dest = dest_dir + '/' + basename(policy_image)
            value_dest = dest_dir + '/' + basename(value_image)
            logger.info("Copying {} to {}".format(policy_image, policy_dest))
            logger.info("Copying {} to {}".format(value_image, value_dest))

            copyfile(policy_image, policy_dest)
            copyfile(value_image, value_dest)


def copy_data_files(data_files, base_dir):
    for problem_name in data_files:
        for mdp in data_files[problem_name]:
            mdp_files = data_files[problem_name][mdp]

            dest_dir = base_dir + '/' + problem_name

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            for file_type in mdp_files:
                file_name = mdp_files[file_type]
                file_dest = dest_dir + '/' + basename(file_name)

                logger.info("Copying {} file from {} to {}".format(file_type, file_name, file_dest))

                copyfile(file_name, file_dest)


def plot_data(data_files, envs, base_dir):
    for problem_name in data_files:
        for mdp in data_files[problem_name]:
            env = lookup_env_from_mdp(envs, mdp)
            if env is None:
                logger.error("Unable to find env for MDP {}".format(mdp))
                return

            mdp_files = data_files[problem_name][mdp]

            step_term = 'Steps'
            if problem_name == 'Q':
                step_term = 'Episodes'

            df = pd.read_csv(mdp_files['file'])

            title = '{}: {} - Time vs {}'.format(env['readable_name'],
                                                 problem_name_to_descriptive_name(problem_name), step_term)
            file_name = '{}/{}/{}_time.png'.format(base_dir, problem_name, mdp)
            p = plot_time_vs_steps(title, df, xlabel=step_term)
            p = watermark(p)
            p.savefig(file_name, format='png', dpi=150)
            p.close()

            reward_term = 'Reward'
            if problem_name in ['VI', 'PI']:
                reward_term = 'Value'

            title = '{}: {} - {} and Delta vs {}'.format(env['readable_name'],
                                                         problem_name_to_descriptive_name(problem_name),
                                                         reward_term, step_term)
            file_name = '{}/{}/{}_reward_delta.png'.format(base_dir, problem_name, mdp)
            p = plot_reward_and_delta_vs_steps(title, df, ylabel=reward_term, xlabel=step_term)
            p = watermark(p)
            p.savefig(file_name, format='png', dpi=150)
            p.close()

            if problem_name == 'Q' and 'episode_file' in mdp_files:
                title = '{}: {} - {}'.format(env['readable_name'], problem_name_to_descriptive_name(problem_name),
                                             '{}')
                episode_df = pd.read_csv(mdp_files['episode_file'])
                q_length, q_reward, q_time = plot_episode_stats(title, episode_df)
                file_base = '{}/{}/{}_{}.png'.format(base_dir, problem_name, mdp, '{}')

                logger.info("Plotting episode stats with file base {}".format(file_base))
                q_length.savefig(file_base.format('episode_length'), format='png', dpi=150)
                q_reward.savefig(file_base.format('episode_reward'), format='png', dpi=150)
                q_time.savefig(file_base.format('episode_time'), format='png', dpi=150)
                plt.close()


def lookup_env_from_mdp(envs, mdp):
    for env in envs:
        if env['name'] == mdp:
            return env

    return None


def problem_name_to_descriptive_name(problem_name):
    if problem_name == 'VI':
        return 'Value Iteration'
    if problem_name == 'PI':
        return 'Policy Iteration'
    if problem_name == 'Q':
        return "Q-Learner"
    return 'Unknown'


def plot_results(envs):
    best_params = {}
    best_images = {}
    data_files = {}
    for problem_name in TO_PROCESS:
        logger.info("Processing {}".format(problem_name))

        problem = TO_PROCESS[problem_name]
        problem_path = '{}/{}'.format(INPUT_PATH, problem['path'])
        problem_image_path = '{}/images/{}'.format(INPUT_PATH, problem['path'])

        best_params[problem_name] = find_optimal_params(problem_name, problem_path, problem['file_regex'])
        best_images[problem_name] = find_policy_images(problem_image_path, best_params[problem_name])
        data_files[problem_name] = find_data_files(problem_path, best_params[problem_name])

    copy_best_images(best_images, REPORT_PATH)
    copy_data_files(data_files, REPORT_PATH)
    plot_data(data_files, envs, REPORT_PATH)
    params_df = pd.DataFrame(best_params)
    params_df.to_csv('{}/params.csv'.format(REPORT_PATH))
