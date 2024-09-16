import yaml
import numpy as np
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, RecurrentPPO
import distutils

# Loads in an experiment config file
def load_experiment(path):
    with open(path, 'r') as experiment_file:
        config = yaml.load(experiment_file, Loader=yaml.FullLoader)
        config['field'] = list(map(lambda x: tuple(x), config['field']))
        config['init_positions'] = list(map(lambda x: np.array(x), config['init_positions']))
        config['infected_locations'] = list(map(lambda x: tuple(x), config['infected_locations']))
    return config

# Loads in a trained model
def load_model(algorithm, st):
    model_path = f'trained_models/{algorithm}_set{st}.zip'
    if algorithm == 'A2C':
        model = A2C.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'PPO':
        model = PPO.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'TRPO':
        model = TRPO.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    else:
        model = RecurrentPPO.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    return model

# Converts a list of binary digits to its decimal equivalent
def binary_list_to_decimal(bin_list):
    bin = ''
    for b in bin_list:
        bin += str(b)
    dec = int(bin,2)
    return dec

# Parses a string into a bool
def parse_bool(string):
    return bool(distutils.util.strtobool(string))
