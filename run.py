import argparse
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, RecurrentPPO
import pygame
import gymnasium as gym
import distutils
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.sim import DroneSimulator
from src.utils import load_experiment

if __name__ == '__main__':

    # Parse arguments
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'RecurrentPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', type=int, required=True, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--simulate', type=parse_bool, default=False, help='If true, uses the Coppelia Simulator to show the environment. If false, renders the environment using PyGame')

    args = parser.parse_args()

    # Load the model
    model_path = f'trained_models/{args.algorithm}_set{args.set}.zip'
    if args.algorithm == 'A2C':
        model = A2C.load(model_path)
    elif args.algorithm == 'PPO':
        model = PPO.load(model_path)
    elif args.algorithm == 'TRPO':
        model = TRPO.load(model_path)
    else:
        model = RecurrentPPO.load(model_path)

    # Make the environment
    env_config = load_experiment(f'experiments/set{args.set}.yaml')
    env = gym.make('ThreeAgentGridworld-v0', render_mode='human', env_config=env_config)
    env.metadata['render_fps'] = 5 if args.simulate else 30
    obs, info = env.reset()

    # Set up CoppeliaSim
    if args.simulate:
        client = RemoteAPIClient()
        print('1')
        sim = client.getObject('sim')
        print('2')
        defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
        sim.setInt32Param(sim.intparam_idle_fps, 0)

        drone_simulator = DroneSimulator(sim, polygon=env.poly_vertices, scaling_factor=5, height=0.35)
        drone_simulator.draw_field()
        drone_simulator.set_agent_positions(k=3, info=info)
        drone_simulator.set_weed_locations(weed_locations=env.infected_locations)
        drone_simulator.start_simulation()

    # Run trained model
    terminated, truncated = False, False
    total_rewards = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(list(action))
        if args.simulate:
            drone_simulator.move_agents(k=3, info=info)
        else:
            env.render()
            pygame.event.get()
        total_rewards += reward
        print(f"Obs: {obs}, Reward: {reward}, terminated: {terminated}, total_rewards: {total_rewards}, action: {action}")
    print('terminated:', terminated, 'truncated:', truncated)
    
    # Close simulator and environment
    if args.simulate:
        drone_simulator.stop_simulation()
    env.close()