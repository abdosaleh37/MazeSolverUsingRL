import numpy as np

class PolicyGradientAgent:
    def __init__(self, n_states, n_actions, lr=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.policy = np.ones((n_states, n_actions)) / n_actions
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def get_action(self, state):
        probs = self.policy[state]
        action = np.random.choice(self.n_actions, p=probs)
        self.episode_states.append(state)
        self.episode_actions.append(action)
        return action

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def learn(self, gamma=0.99):
        G = 0
        returns = []
        for r in reversed(self.episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for state, action, Gt in zip(self.episode_states, self.episode_actions, returns):
            baseline = self.policy[state]
            grad = np.eye(self.n_actions)[action] - baseline
            self.policy[state] += self.lr * Gt * grad
            self.policy[state] = np.clip(self.policy[state], 1e-8, 1)
            self.policy[state] /= self.policy[state].sum()

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [] 