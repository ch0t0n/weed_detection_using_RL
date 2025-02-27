import os
import argparse
from datetime import datetime
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO, ARS, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from src.utils import load_experiment, load_model, parse_bool

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'DQN', 'ARS', 'RecurrentPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor for the DRL algorithm')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')
    
    args = parser.parse_args()
    print(args)
    
    # Configure environment
    env_config = load_experiment(f'experiments/set{args.set}.yaml')
    vec_env = make_vec_env('ThreeAgentGridworld-v1', env_kwargs={'env_config': env_config, 'seed': args.seed}, n_envs=4)
    vec_env.seed(seed=args.seed)
    vec_env.action_space.seed(seed=args.seed)
    
    os.makedirs('logs', exist_ok=True)

    # Configure model
    if args.resume:
        model = load_model(args.algorithm, args.set, args.seed)
        model.set_env(vec_env)
    else:
        model_args = {
            'policy': 'MlpLstmPolicy' if args.algorithm == 'RecurrentPPO' else 'MlpPolicy',
            'env': vec_env,
            'verbose': args.verbose,
            'tensorboard_log': './logs',
            'seed': args.seed,
            'device': args.device,
        }
        if args.algorithm != 'ARS':
            model_args['gamma'] = args.gamma

        if args.algorithm == 'A2C':
            model = A2C(**model_args)
        elif args.algorithm == 'PPO':
            model = PPO(**model_args)
        elif args.algorithm == 'TRPO':
            model = TRPO(**model_args)
        elif args.algorithm == 'DQN':
            model = DQN(**model_args)
        elif args.algorithm == 'ARS':
            model = ARS(**model_args)
        else:
            model = RecurrentPPO(**model_args)

    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    logger = LogEveryNTimesteps(n_steps=args.log_steps)
    model.learn(total_timesteps=args.steps, callback=logger, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')
    
    # Save model
    os.makedirs('trained_models', exist_ok=True)
    model.save(f'trained_models/{args.algorithm}_set{args.set}.zip')

    vec_env.close()
