"""
This file is responsible for running the game against 
mvps 5 times and collecting the energy_gain which is 
defined as:
    energy_gain = player_life - enemy_life
"""

from typing import List
import numpy as np
import pickle
import os
from ea import get_env_for_enemies


def run_mvp(mvp: np.ndarray, enemy: int, num_runs: int = 5):
    """
    This function takes an mvp, the number of hidden nodes the mvp 
    has, and an enemy to fight against. 

    It will return a list of resulting energy_gain for each run in num_runs.
    """
    energy_gains = []
    env = get_env_for_enemies([enemy])

    for _ in range(num_runs):
        _, player_life, enemy_life, _ = env.play(pcont=mvp)
        energy_gains.append(player_life - enemy_life)

    return energy_gains


def load_mvps(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_mvps_for_ea_and_enemy(data_dir: str, enemy: int, ea: int):
    file_path = os.path.join(data_dir, f"enemy.{enemy}.ea.{ea}.mvps.pickle")
    return load_mvps(file_path)


def gather_mvp_stats_for_ea_and_enemy(data_dir: str, enemy: int, ea: int):
    """
    Here, we will load some mvps from a pickle file and run each mvp 5x against
    the enemy, gathering the resulting energy_gains. Each MVP will have their 
    energy_gains averaged, and added to a result array. The resulting array of
    means per MVP is returned.
    """
    means = []
    mvps = load_mvps_for_ea_and_enemy(data_dir, enemy, ea)
    for mvp in mvps:
        gains = run_mvp(mvp, enemy)
        means.append(np.mean(gains))
    return means


def save_mvp_run(data_dir: str, enemy: int, ea: int, result: List[float]):
    f_name = os.path.join(data_dir, f'enemy.{enemy}.ea.{ea}.mvp_run.pickle')
    print(f"Saving data to {f_name}")
    with open(f_name, 'wb') as f:
        pickle.dump(result, f)


def gather_data(data_dir: str):
    for ea in [1, 2]:
        for enemy in [1, 2, 8]:
            print(f"Gathering stats for ea {ea} and enemy {enemy}")
            results = gather_mvp_stats_for_ea_and_enemy(data_dir, enemy, ea)
            save_mvp_run(data_dir, enemy, ea, results)
            print(f'Results for enemy {enemy} and ea {ea}: ', results)


def main():
    data_dir = 'data_dir'
    gather_data(data_dir)


if __name__ == '__main__':
    main()