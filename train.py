import argparse
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from src.utils import load_experiment

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'RecurrentPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor for the DRL algorithm')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    
    args = parser.parse_args()
    
    env_config = load_experiment(f'experiments/set{args.set}.yaml')
    vec_env = make_vec_env('ThreeAgentGridworld-v0', env_kwargs={'env_config': env_config}, n_envs=4)
    
    if args.algorithm == 'A2C':
        model = A2C("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
    elif args.algorithm == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
    elif args.algorithm == 'TRPO':
        model = TRPO("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
    else:
        model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)

    model.learn(total_timesteps=args.steps)
    model.save(f'trained_models/{args.algorithm}_set{args.set}.pth')

    vec_env.close()
