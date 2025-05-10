import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, action_space, state_size, epsilon=0.02, alpha=0.5, gamma=0.95):
        """
        Initialize the Q-learning agent with the necessary parameters.

        Parameters:
        - action_space (int): The number of possible actions in the environment.
        - state_size (int): The number of possible states.
        - epsilon (float): The exploration rate (probability of choosing a random action).
        - alpha (float): The learning rate (how much new information overrides old information).
        - gamma (float): The discount factor (how much future rewards are valued).
        """
        self.action_space = action_space    # Number of actions
        self.epsilon = epsilon              # Exploration rate
        self.alpha = alpha                  # Learning rate
        self.gamma = gamma                  # Discount factor
        
        # Initialize the Q-table with zeros; it has one row per state and one column per action
        self.q_table = np.zeros((state_size, action_space))

    def get_action(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.

        Parameters:
        - state (int): The current state of the agent.

        Returns:
        - action (int): The selected action.
        """
        # Exploration: Choose a random action with probability epsilon
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            # Exploitation: Choose the action with the highest Q-value for the current state
            if state >= len(self.q_table):  
                # Handle out-of-bounds state (in case state is not in Q-table range)
                return np.random.choice(self.action_space)
            else:
                # Return the action with the max Q-value
                return np.argmax(self.q_table[state])


    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the reward received from taking the action in the given state.

        Parameters:
        - state (int): The current state of the agent.
        - action (int): The action taken.
        - reward (float): The reward received for the action.
        - next_state (int): The state reached after taking the action.
        - done (bool): Whether the episode has ended (True or False).
        """
        # Current Q-value for the given state-action pair
        current_q = self.q_table[state, action]
        
        # Max Q-value for the next state (used in the Bellman equation)
        max_next_q = np.max(self.q_table[next_state])

        # If done (episode ends), the new Q-value is just the reward
        if done:
            new_q = reward
        else:
            # Otherwise, calculate the updated Q-value using the Bellman equation
            new_q = reward + self.gamma * max_next_q
        
        # Update the Q-value for the current state-action pair using a weighted average
        self.q_table[state, action] = (1 - self.alpha) * current_q + self.alpha * new_q
        
    # Save the Q-table i a file
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    # Load the Q-table in a file
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
