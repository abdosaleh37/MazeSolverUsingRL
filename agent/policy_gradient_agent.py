import numpy as np

class PolicyGradientAgent:
    def __init__(self, n_states, n_actions, lr=0.1):
        """
        Initialize the Policy Gradient Agent with necessary parameters.

        Parameters:
        - n_states (int): Number of states in the environment.
        - n_actions (int): Number of possible actions.
        - lr (float): Learning rate for policy updates.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        
        # Initialize policy as a uniform distribution (equally likely to choose each action in each state)
        self.policy = np.ones((n_states, n_actions)) / n_actions
        
        # Initialize empty lists to store episode information (states, actions, rewards)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def get_action(self, state):
        """
        Given a state, sample an action from the policy distribution.

        Parameters:
        - state (int): The current state of the agent.

        Returns:
        - action (int): The chosen action based on the policy.
        """
        # Get the action probabilities from the policy for the given state
        probs = self.policy[state]
        
        # Sample an action from the action space based on the action probabilities
        action = np.random.choice(self.n_actions, p=probs)
        
        # Store the current state and action for the learning phase
        self.episode_states.append(state)
        self.episode_actions.append(action)
        
        return action

    def store_reward(self, reward):
        """
        Store the reward obtained from the environment for the last action.

        Parameters:
        - reward (float): The reward obtained for the previous action.
        """
        self.episode_rewards.append(reward)

    def learn(self, gamma=0.99):
        """
        Perform the policy update using the collected rewards and actions over an episode.

        Parameters:
        - gamma (float): Discount factor for future rewards (default 0.99).
        """
        G = 0
        returns = []
        
        # Compute the returns (discounted sum of future rewards) for each time step
        for r in reversed(self.episode_rewards):
            G = r + gamma * G       # Discounted reward
            returns.insert(0, G)    # Insert the return at the beginning of the list
            
        # Convert returns to a numpy array and normalize them for stability
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)   # Normalize returns

        # Update the policy for each state-action pair
        for state, action, Gt in zip(self.episode_states, self.episode_actions, returns):
            baseline = self.policy[state]   # Current policy probabilities for the state
            grad = np.eye(self.n_actions)[action] - baseline
            
            # Update the policy based on the gradient of the log-probability of the action
            self.policy[state] += self.lr * Gt * grad
            
            # Clip the policy probabilities to avoid extreme values and ensure they remain valid
            self.policy[state] = np.clip(self.policy[state], 1e-8, 1)
            
            # Normalize the policy to sum to 1 (valid probability distribution)
            self.policy[state] /= self.policy[state].sum()

        # Reset episode data after learning
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [] 