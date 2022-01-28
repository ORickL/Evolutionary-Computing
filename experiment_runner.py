"""
This file is responsible for running an experiment using a deap toolbox instance and
collecting the results and saving them to files in a reasonable way.
"""

import os, csv
from pathlib import Path
from datetime import datetime
from deap.base import Toolbox
from deap.algorithms import eaSimple
from deap.tools.support import HallOfFame
from numpy import array_equal

# Local imports
from stats import RunStats

os.environ["SDL_VIDEODRIVER"] = "dummy"


def save_individual(individual, filename):
    with open(filename, 'w') as f:
        for weight in individual:
            f.write(f"{weight}\n")


def save_stats_logs(stats_logs, filename):
    keys = stats_logs[0].keys()
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(stats_logs)


def  run_experiment(*, 
        experiment_name: str, 
        toolbox: Toolbox, 
        cxpb: float, 
        mutpb: float, 
        ngen: int, 
        **kwargs):
    """
    This is the main export of the file. This will run:
        1. Create a folder to store the data files
        2. Create a population using toolbox.population()
        3. Instantiate stats collector
        4. Run EA
        5. Save stats file
        6. Save best individual in mvp file

    For each run we will need to collect:
    1. mean and max fitness for each generation (will be saved in a file called genstats.csv)
    2. the best individual from the experiment (saved to a file called mvp.txt)
    """
    dir = _get_or_create_folder(experiment_name)
    population = toolbox.population(**kwargs)
    stats = RunStats()
    hof = HallOfFame(1, similar=array_equal)
    result, logs = eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, halloffame=hof)
    mvp = hof.items[0]

    # Save MVP to file
    mvp_file = (Path(dir) / "mvp.txt").resolve()
    save_individual(mvp, mvp_file)

    # Save stats logs
    stats_file = (Path(dir) / "genstats.csv").resolve()
    save_stats_logs(logs, stats_file)

    return mvp, logs

def _get_or_create_folder(experiment_name: str) -> str:
    dir = _get_folder(experiment_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def _get_folder(experiment_name: str):
    return (Path(".") / "data" / experiment_name).resolve()
