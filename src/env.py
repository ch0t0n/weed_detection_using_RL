import copy
import pygame
import gymnasium as gym
from gymnasium import spaces
from shapely import Polygon
from shapely.geometry import Point
import numpy as np
from src.utils import binary_list_to_decimal, decode_action
from itertools import product

# The agricultural field environment
class ThreeAgentGridworldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 4}    
    def __init__(self, seed=None, render_mode=None, env_config=None):
        super(ThreeAgentGridworldEnv, self).__init__()
        self.config = copy.deepcopy(env_config)
        self.poly_vertices = self.config['field']
        self.Poly = Polygon(self.poly_vertices) # Get the points of the polygon
        self.window_size = 800  # The size of the PyGame window        
        self.grid_size = self.config['grid_size'] # Size of the grid
        self.outer_boundary = self.Poly.buffer(distance=2) # outer boundary with buffer of distance 2

        # Observation points
        self.observation_points = self.obs_points()
        self.observation_length = len(self.observation_points)
        self.observation_map = {tuple(v):i for i,v in enumerate(self.observation_points)}

        # Keep track of visited states and count steps
        self.step_count = 0
        self.visited = set()
        self.infected_locations = self.config['infected_locations']
        
        self.infected_length = len(self.infected_locations)
        self.infected_state_length = 2**10 # 10 weeds max, binary to decimal
        self.infected_dict = {v:0 for v in self.infected_locations} # dictionary of locations
        
        # Action and observation space
        self.action_space = spaces.Discrete(5 ** 3)  # 5 possible actions for 3 agents
        self.observation_space = spaces.MultiDiscrete([self.observation_length, self.observation_length, self.observation_length, self.infected_state_length])
        
        assert render_mode is None or render_mode in self.metadata["render_modes"] # Check if the render mode is correct
        self.render_mode = render_mode
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        # Reset the environment and start
        self.reset(seed=seed)

    def obs_points(self):
        xs = np.arange(0, 100, 1)
        ys = np.arange(0, 100, 1)
        obs_points = list(product(xs, ys))
        return obs_points
    
    def _get_obs(self):
        a1, a2, a3 = self.agent_positions[0], self.agent_positions[1], self.agent_positions[2]
        info = {'agent1': a1, 'agent2': a2, 'agent3': a3, 'step_count': self.step_count}
        p1,p2,p3 = self.observation_map[tuple(a1)], self.observation_map[tuple(a2)], self.observation_map[tuple(a3)]
        infected = binary_list_to_decimal(list(self.infected_dict.values()))
        state = np.array([p1,p2,p3,infected]) # convert the infected binary list to decimal
        return state, info

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.visited = set()
        self.step_count = 0
        self.infected_locations = copy.deepcopy(self.config['infected_locations'])
        self.infected_dict = {v:0 for v in self.infected_locations} # 0 for unvisited infected locations, 1 for visited
        self.agent_positions = copy.deepcopy(self.config['init_positions'])
        return self._get_obs()

    def step(self, action):
        # Placeholder for terminal state and rewards
        terminated, truncated = False, False
        rewards = 0
        self.step_count += 1

        # Define the movements corresponding to each action
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, none
        
        # Update the positions of both agents
        decoded_action = decode_action(action)
        for i, act in enumerate(decoded_action):

            movement = movements[act] # What movement to take
            new_position = self.agent_positions[i] + movement # New position after movement
            
            # Ensure the new position is within bounds
            new_p = Point(new_position[0], new_position[1])
            if self.Poly.contains(new_p):
                self.agent_positions[i] = new_position
            else:
                rewards -= 10
            if tuple(new_position) in self.visited:
                rewards -= 10
            else:
                rewards -= 1
            self.visited.add(tuple(new_position))
        
        # Check if an infected location is visited
        # TODO: Should we make a dedicated action for removing the weed instead of doing it automatically?
        # TODO: Perhaps adding a cost of removing the weed since real drones will have limited herbicide and should be discouraged from wasting it
        infected_visited = [x for x in self.agent_positions if tuple(x) in self.infected_locations] # If infected cells are visited
        for v in infected_visited:
            if tuple(v) in self.infected_locations:
                self.infected_locations.remove(tuple(v))
                self.infected_dict[tuple(v)] = 1
        
        if infected_visited:
            rewards += 100 * len(infected_visited) 

        if sum(list(self.infected_dict.values()))==self.infected_length:
            rewards += 100000
            terminated = True

        
        # If the agents meet at the same position, we can assign a reward or consider it a terminal state
        if np.array_equal(self.agent_positions[0], self.agent_positions[1]) or np.array_equal(self.agent_positions[0], self.agent_positions[2]) or np.array_equal(self.agent_positions[1], self.agent_positions[2]):
            rewards -= 100000  # Infinity reward for meeting at the same position
            terminated = True
        
        obs, info = self._get_obs()
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == 'print':
            grid = np.zeros((self.grid_size, self.grid_size))
            grid[tuple(self.agent_positions[0])] = 1  # Mark the position of the first agent
            grid[tuple(self.agent_positions[1])] = 2  # Mark the position of the second agent
            print(grid)
        else:
            if self.window is None and self.render_mode == "human": # Initialize pygame if it is not initialized
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()
            
            # Fill the canvas
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            pix_square_size = (
                self.window_size / self.grid_size
            )  # The size of a single grid square in pixels

            # Draw the polygon
            pixel_poly_vertices = [(point[0] * pix_square_size, point[1] * pix_square_size) for point in self.poly_vertices]
            pygame.draw.polygon(surface=canvas, 
                                color=(255, 255, 0), 
                                points=pixel_poly_vertices)
            
            # Draw the visited regions
            for p in self.visited:
                pygame.draw.rect(
                canvas,
                pygame.Color(100, 100, 100, a=0.5),
                pygame.Rect(
                    pix_square_size * np.array(p),
                    (pix_square_size, pix_square_size),
                ),
                )
            # Draw agent1 (square)
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self.agent_positions[0],
                    (pix_square_size, pix_square_size),
                ),
            )
            # Draw agent2 (circle)
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self.agent_positions[1] + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            # Draw agent3 (circle)
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (self.agent_positions[2] + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            # Draw infected locations
            for l in self.infected_locations:
                pygame.draw.rect(
                    canvas,
                    (0, 255, 255),
                    pygame.Rect(
                        pix_square_size * np.array(l),
                        (pix_square_size, pix_square_size),
                    ),
                )


            if self.render_mode == "human":
                # The following line copies our drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
                # Finally
                pygame.event.get()

            elif self.render_mode == 'rgb_array':  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
