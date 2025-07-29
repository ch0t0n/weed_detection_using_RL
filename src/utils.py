import yaml
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO, ARS, RecurrentPPO
import distutils
import inspect

# Loads in an experiment config file
def load_experiment(path):
    with open(path, 'r') as experiment_file:
        config = yaml.load(experiment_file, Loader=yaml.FullLoader)
        config['field'] = list(map(lambda x: tuple(x), config['field']))
        config['init_positions'] = list(map(lambda x: np.array(x), config['init_positions']))
        config['infected_locations'] = list(map(lambda x: tuple(x), config['infected_locations']))
    return config

# Loads in a trained model
def load_model(algorithm, experiment_set, seed, device, models_dir, verbose, log_dir):
    model_args = {
        'path': f'{models_dir}/{algorithm}_set{experiment_set}.zip',
        'tb_log_name': f'{algorithm}_set{experiment_set}',
        'device': device,
        'seed': seed,
        'verbose': verbose,
        'tensorboard_log': log_dir,
    }

    if algorithm == 'A2C':
        model = A2C.load(**model_args)
    elif algorithm == 'PPO':
        model = PPO.load(**model_args)
    elif algorithm == 'TRPO':
        model = TRPO.load(**model_args)
    elif algorithm == 'DQN':
        model = DQN.load(**model_args)
    elif algorithm == 'ARS':
        model = ARS.load(**model_args)
    else:
        model = RecurrentPPO.load(**model_args)
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

# Encoding function: (x, y, z) → Discrete
def encode_action(action):
    x, y, z = action
    return x + 5 * y + 25 * z  # 25 = 5 * 5

# Decoding function: Discrete → (x, y, z)
def decode_action(action):
    x = action // 25
    y = (action // 5) % 5
    z = action % 5
    return np.array([x, y, z])

# Filters out arguments that are not present in a model's constructor
def filter_args(args, model):
    model_kwargs = inspect.getfullargspec(model).args
    return {k:args[k] for k in args if k in model_kwargs}
