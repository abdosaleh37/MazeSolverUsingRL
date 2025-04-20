import gym
from gym import spaces
import numpy as np
from .utils import generate_maze, render_maze

class MazeEnv(gym.Env):
    def __init__(self, maze=None, size=(10, 10)):
        self.size = size
        self.maze = np.array(maze if maze is not None else generate_maze(size))
        
        self.start = (0, 0)
        self.goal = (size[0]-1, size[1]-1)
        self.agent_position = self.start
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(size[0] * size[1])

    def _get_state(self):
        x, y = self.agent_position
        return x * self.size[1] + y

    def reset(self):
        self.agent_position = self.start
        return self._get_state()

    def step(self, action):
        x, y = self.agent_position

        if action == 0: x -= 1  # UP
        elif action == 1: x += 1  # DOWN
        elif action == 2: y -= 1  # LEFT
        elif action == 3: y += 1  # RIGHT

        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and self.maze[x, y] == 0:
            self.agent_position = (x, y)

        reward = -1
        done = False
        if self.agent_position == self.goal:
            reward = 100
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        render_maze(self.maze, self.agent_position, self.goal)
