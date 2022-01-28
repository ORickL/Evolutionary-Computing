from typing import List, Tuple
from deap.tools.support import Logbook
from scipy.stats import ttest_ind
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

nruns = 5
directory = "data_dir"

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_pickle_data(enemy, ea, data_type):
    """
    Get pickle data from a file in the directory <directory>.
    """

    file_name = f"enemy.{enemy}.ea.{ea}.{data_type}.pickle"

    with open(f"{directory}/{file_name}", "rb") as f:

        obj = pickle.load(f)

    return obj

def add_results_to_plot(ax: plt.Axes, logbooks: List[Logbook], name: str, color: str):
    # Get mean and max values from the logbooks for each generation for each run
    means = np.array([logbook.select('mean') for logbook in logbooks])
    maxs = np.array([logbook.select('max') for logbook in logbooks])
    
    # Calculate the standard deviation for both the mean value and the max value for each generation
    means_std_dev = np.std(means, axis=0)
    maxs_std_dev = np.std(maxs, axis=0)

    # Calculate the mean mean value and the mean max value for each generation
    mean_mean = means.mean(axis=0)
    mean_max =  maxs.mean(axis=0)

    ngens = len(means[0])
    gens = list(range(ngens))
    ax.fill_between(gens, (mean_mean-0.5*means_std_dev), (mean_mean+0.5*means_std_dev), color=color, alpha=.1)
    ax.plot(gens, mean_mean, label=f"{name} Mean value", color=color)

    ax.fill_between(gens, (mean_max-0.5*maxs_std_dev), (mean_max+0.5*maxs_std_dev), color=color, alpha=.1)
    ax.plot(gens, mean_max, '--', label=f"{name} Max value", color=color)

def create_box_plot(ea1_mvp_runs, ea2_mvp_runs, enemy):
    
    fig, ax = plt.subplots()

    box_plot_data = [ea1_mvp_runs, ea2_mvp_runs]
    box_plot_labels = ["Lévy", "Gauss"]

    ax.set_ylabel("Gain")
    ax.boxplot(box_plot_data, labels=box_plot_labels)
    ax.set_title(f"MVP performance for enemy {enemy}")

    plt.tight_layout()
    plt.savefig(f"fig_dir/boxplot_enemy{enemy}")
    # plt.show()

def plot_logbooks(ea1_logbooks: List[Logbook], ea2_logbooks: List[Logbook], enemy):
    """
    This function just makes plots given a set of logbooks
    """
    
    # Plot the data
    fig, ax = plt.subplots()

    add_results_to_plot(ax, ea1_logbooks, 'Lévy', 'b')
    add_results_to_plot(ax, ea2_logbooks, 'Gauss', 'r')
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Improvement of Algorithms on enemy {enemy}")

    ax.legend() 
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'fig_dir/enemy{enemy}.png')
    # plt.show()

def get_ttest(data_dir: str, enemy: int, eas: Tuple[int, int]) -> Tuple[float, float]:
    f_names = map(lambda ea: os.path.join(data_dir, f'enemy.{enemy}.ea.{ea}.mvp_run.pickle'), eas)
    datasets = []
    for f_name in f_names:
        with open(f_name, 'rb') as f:
            datasets.append(pickle.load(f))
    d1, d2 = datasets
    result = ttest_ind(d1, d2)
    return result.statistic, result.pvalue

  
def main():
    
    enemies = [2,5,8]
    for enemy in enemies:

        ea1_enemy_logs = get_pickle_data(enemy, 1, "logs")
        ea2_enemy_logs = get_pickle_data(enemy, 2, "logs")
        plot_logbooks(ea1_enemy_logs, ea2_enemy_logs, enemy)

    for enemy in enemies:
        
        print(get_ttest(directory, enemy, (1, 2)))
        ea1_mvp_runs = get_pickle_data(enemy, 1, "mvp_run")
        ea2_mvp_runs = get_pickle_data(enemy, 2, "mvp_run")
        create_box_plot(ea1_mvp_runs, ea2_mvp_runs, enemy)

    plt.show()
    
if __name__ == "__main__":
    main()
