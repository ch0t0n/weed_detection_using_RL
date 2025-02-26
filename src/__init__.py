import gymnasium as gym
from src.env import ThreeAgentGridworldEnv

# Register the environment
gym.envs.registration.register(
    id='ThreeAgentGridworld-v1',
    entry_point=ThreeAgentGridworldEnv,
    max_episode_steps=2000,
)
