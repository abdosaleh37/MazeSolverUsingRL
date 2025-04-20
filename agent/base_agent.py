class BaseAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError