import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, action_space, state_size, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((state_size, action_space.n))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space.n)
        else:
            if state >= len(self.q_table):  
                return np.random.choice(self.action_space.n)
            else:
                return np.argmax(self.q_table[state])


    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]

        max_next_q = np.max(self.q_table[next_state])

        if done:
            new_q = reward
        else:
            new_q = reward + self.gamma * max_next_q
        self.q_table[state, action] = (1 - self.alpha) * current_q + self.alpha * new_q
        
        
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)


    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
