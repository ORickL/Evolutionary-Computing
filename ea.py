import pickle
import re
from typing import List, Optional
from numpy import ndarray, log
from deap import base, creator, tools
from random import uniform
from experiment_runner import run_experiment
from custom_mutation import mutLevy
from my_controller import my_controller

import os, sys
sys.path.insert(0,'evoman')
from environment import Environment

# Model tuning
n_hidden = 10
pop_size = 50
cxpb=0.5
mutpb=0.1
ngen=10
len_inputs = 20
num_output = 5

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Our fitness function will be just to try to maximize fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Our individual will just be the single set of hidden nodes
creator.create("Individual", ndarray, fitness=creator.FitnessMax)

class MyEnv(Environment):
    def fitness_single(self):
       return 0.99*(100 - self.get_enemylife()) + 0.01*self.get_playerlife() - log(self.get_time())


def get_env_for_enemies(enemies: List[int]):
    return MyEnv(experiment_name='experiment',
            enemymode="static",
            fullscreen=False,
            use_joystick=False,
            playermode='ai',
            enemies=enemies,
            randomini='yes',
            player_controller=my_controller(n_hidden),
            logs="off")


def evaluate(individual, env=None):
    """
    Here we will run a game simulation with the given encoding of the individual.
    The encoding is a binary array, with n_hidden layers. Therefore, we can use
    the built in demo player_controller out of the box?

    We actually need to encode both the first and second layer of the model into the 1-dimensional array. 
    We need the weights of each of the connections between inputs (len_inputs) and hidden nodes (n_hidden)
    (len_inputs * n_hidden) as the first part of the Individual.

    Then the second part of the array will be each of the connections in the hidden layer to output layers,
    so (n_hidden * num_output).

    The actual fitness will just come from playing the game, so play the game and
    then return the reported fitness. EZPZ
    """
    fitness, _, _, _ = env.play(pcont=individual)
    return (fitness,)


"""
first we have biases for the n_hidden
Then we have connections between len_inputs * n_hidden
then we have biases for the output layer of size num_output
Then we have connections between n_hidden * num_output
"""
individual_size = n_hidden + (len_inputs * n_hidden) + num_output + (n_hidden * num_output)


def get_base_toolbox(env: Environment):
    # Make our toolbox to run our ea algorithm given an environment
    toolbox = base.Toolbox()
    toolbox.register('coinflip', uniform, -1, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.coinflip, n=individual_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=pop_size)
    toolbox.register('evaluate', evaluate, env=env)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('select', tools.selTournament, tournsize=int(pop_size / 3))
    return toolbox


def get_levy_toolbox(env: Environment):
    toolbox = get_base_toolbox(env)
    toolbox.register('mutate', mutLevy, mu=0., c=.5, indpb=mutpb)
    return toolbox


def get_gaussian_toolbox(env: Environment):
    toolbox = get_base_toolbox(env)
    toolbox.register('mutate', tools.mutGaussian, mu=0., sigma=.5, indpb=mutpb)
    return toolbox


def do_assignment(
        data_dir: str,
        experiment_name: str, 
        cxpb: float, 
        mutpb: float, 
        ngen: int, 
        load_dir: Optional[str] = None,
        **kwargs):
    """
    To do the assignment, we have to do 2 EA's, represented by the 2 toolboxes.
    The things we have to do for the assignemnt per EA is:
        Loop for i in 3 enemies ([1, 2, 8]):
            Loop for j in range(10):
                1. ea1_mvp, ea1_result, ea1_logbook = Run the EA1 on enemy[i]
                2. ea2_mvp, ea2_result, ea2_logbook = Run the EA2 on enemy[i]
                3. plot logbooks
                4. with the mvps:
                    gain_mean1 = mean(gain(mvp1) for k in range(5))
                    gain_mean2 = mean(gain(mvp2) for k in range(5))
                    boxplot(gain_mean)    

    We will save data and individuals frequently.
    """
    enemies = [6, 7]
    num_iters = 10
    num_mvp_runs = 5
    runs_by_enemy = collect_training_iters(data_dir, 
            enemies,
            num_iters,
            experiment_name=experiment_name,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            **kwargs)\
        if load_dir is None \
        else load_training_runs(load_dir)
    breakpoint()


def collect_training_iters(data_dir: str,
        enemies: List[int],
        num_iters: int = 10,
        save_runs: bool = True,
        experiment_name: str = 'experiment',
        **kwargs):
    """
    Runs the experiments and saves to data_dir. Returns list of tuples of
    [(ea1_mvps, ea1_logbooks, ea2_mvps, ea2_logbooks)], 1 tuple for each enemy.
    """
    runs = []
    
    for enemy in enemies:
        ea1_mvps = []
        ea2_mvps = []

        ea1_logbooks = []
        ea2_logbooks = []
        
        env = get_env_for_enemies([enemy])

        levy_toolbox = get_levy_toolbox(env)
        gauss_toolbox = get_gaussian_toolbox(env)

        # breakpoint()

        for iter in range(num_iters):
            print(f"..................\nRunning EA1 on enemy {enemy}, {iter} iteration\n....................")
            mvp1, logs1 = run_experiment(experiment_name=f"{experiment_name}.iter.{iter}.ea.1", toolbox=levy_toolbox, **kwargs)
            ea1_mvps.append(mvp1)
            ea1_logbooks.append(logs1)

            print(f"..................\nRunning EA2 on enemy {enemy}, {iter} iteration\n....................")
            mvp2, logs2 = run_experiment(experiment_name=f"{experiment_name}.iter.{iter}.ea.2", toolbox=gauss_toolbox, **kwargs)
            ea2_mvps.append(mvp2)
            ea2_logbooks.append(logs2)
        
        if save_runs:
            save_obj(data_dir, f"enemy.{enemy}.ea.1.mvps.pickle", ea1_mvps)
            save_obj(data_dir, f"enemy.{enemy}.ea.2.mvps.pickle", ea2_mvps)
            save_obj(data_dir, f"enemy.{enemy}.ea.1.logs.pickle", ea1_logbooks)
            save_obj(data_dir, f"enemy.{enemy}.ea.2.logs.pickle", ea2_logbooks)

        runs.append((ea1_mvps, ea1_logbooks, ea2_mvps, ea2_logbooks))

    return runs

def load_training_runs(data_dir: str):
    regex = re.compile('enemy\\.(\\d+)\\.ea\\.(\\d+)\\.(.+)\\.pickle')
    files = [((regex.match(f).groups()), f) for f in os.listdir(data_dir) 
            if os.path.isfile(os.path.join(data_dir, f)) 
            and regex.match(f) is not None]
    runs = []
    for (enemy, ea, dt), filename in files:
        pass
    pass
        
def save_obj(dir: str, name: str, o: object):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    p = os.path.join(dir, name)
    print(f"Saving an object to file {p}")
    with open(p, 'wb') as f:
        pickle.dump(o, f)


def main():
    # levy_toolbox = get_levy_toolbox()
    # gauss_toolbox = get_gaussian_toolbox()
    do_assignment('data_dir', 'experiment', cxpb, mutpb, ngen)


if __name__ == '__main__':
    main()
