import gym
from gym import spaces
import numpy as np
from .utils import generate_maze, render_maze  # استيراد الأدوات المساعدة

class MazeEnv(gym.Env):
    def __init__(self, maze=None, size=(10, 10)):
        # إذا ما كانش فيه متاهة، هنولد واحدة عشوائيًا
        self.maze = maze if maze is not None else generate_maze(size)
        self.size = size
        
        self.start = (0, 0)  # نقطة البداية
        self.goal = (size[0]-1, size[1]-1)  # نقطة النهاية
        self.agent_position = self.start
        
        self.action_space = spaces.Discrete(4)  # 4 تحركات: UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Discrete(size[0] * size[1])  # مصفوفة الحالة (بأبعاد المتاهة)
    
    def reset(self):
        self.agent_position = self.start  # إعادة الوكيل لنقطة البداية
        return self.agent_position
    
    def step(self, action):
        # التحركات: UP, DOWN, LEFT, RIGHT
        x, y = self.agent_position
        
        if action == 0:  # UP
            x -= 1
        elif action == 1:  # DOWN
            x += 1
        elif action == 2:  # LEFT
            y -= 1
        elif action == 3:  # RIGHT
            y += 1
        
        # التأكد من أن الوكيل مش هيتجاوز حدود المتاهة أو يقفز فوق الجدران
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and self.maze[x, y] == 0:
            self.agent_position = (x, y)
        
        # تحديد المكافأة
        reward = -1  # كل خطوة تكلفتها -1 (عقاب)
        done = False
        
        if self.agent_position == self.goal:
            reward = 100  # مكافأة إذا وصلنا للنهاية
            done = True
        
        return self.agent_position, reward, done, {}
    
    def render(self):
        render_maze(self.maze, self.agent_position, self.goal)  # استدعاء دالة render من utils.py
