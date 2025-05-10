import gym
from gym import spaces
import numpy as np
from .utils import generate_maze
from .pygame_renderer import PygameMazeRenderer

class MazeEnv(gym.Env):
    """
    Custom Maze Environment built using OpenAI Gym interface.
    The agent starts at the top-left corner and must reach the bottom-right goal 
    while navigating around walls.
    """
    def __init__(self, maze=None, size=(10, 10)):
        #Initialize the environment.
        self.size = size
        self.maze = np.array(maze if maze is not None else generate_maze(size))
        
        self.start = (0, 0)                     # Agent starts at the top left
        self.goal = (size[0]-1, size[1]-1)      # Goal is at the bottom right
        self.agent_position = self.start
        
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, right, left
        self.observation_space = spaces.Discrete(size[0] * size[1])
        
        # Initialize renderer as None - will be created when needed
        self.renderer = None

    def _get_state(self):
        x, y = self.agent_position
        return x * self.size[1] + y

    # Reset the environment to the initial state
    def reset(self):
        self.agent_position = self.start
        return self._get_state()

    # Apply an action to the environment and update the agent's position
    def step(self, action):
        x, y = self.agent_position

        # Move the agent according to the chosen action
        if action == 0: x -= 1      # Up
        elif action == 1: x += 1    # Down
        elif action == 2: y -= 1    # Left
        elif action == 3: y += 1    # Right

        # Check for valid move within bounds and not hitting a wall
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and self.maze[x, y] == 0:
            self.agent_position = (x, y)

        # Step cost
        reward = -1
        done = False
        
        # Check if goal is reached
        if self.agent_position == self.goal:
            # Reward of reaching the goal
            reward = 100
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        """
        Render the current state of the environment using Pygame.
        This visualizes the maze, agent, and goal.
        """
        if self.renderer is None:
            self.renderer = PygameMazeRenderer(self.size)
        return self.renderer.render(self.maze, self.agent_position, self.goal)
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
