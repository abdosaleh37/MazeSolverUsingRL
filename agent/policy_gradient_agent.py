import numpy as np

class PolicyGradientAgent:
    def __init__(self, n_states, n_actions, lr=0.185, lr_decay=0.999, entropy_coef=0.01, min_prob=0.05, temperature=1.0):
        """
        Initialize the Policy Gradient Agent with necessary parameters.

        Parameters:
        - n_states (int): Number of states in the environment.
        - n_actions (int): Number of possible actions.
        - lr (float): Learning rate for policy updates.
        - lr_decay (float): Learning rate decay factor.
        - entropy_coef (float): Entropy regularization coefficient.
        - min_prob (float): Minimum probability for exploration.
        - temperature (float): Temperature parameter for action selection.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.lr_decay = lr_decay
        self.entropy_coef = entropy_coef
        self.min_prob = min_prob
        self.temperature = temperature
        
        # Initialize policy with uniform distribution
        self.policy = np.ones((n_states, n_actions)) / n_actions
        
        # Initialize empty lists to store episode information
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # For reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0

    def get_action(self, state):
        """
        Given a state, sample an action from the policy distribution.

        Parameters:
        - state (int): The current state of the agent.

        Returns:
        - action (int): The chosen action based on the policy.
        """
        probs = self.policy[state]
        # Apply temperature scaling
        probs = np.exp(np.log(probs) / self.temperature)
        # Ensure minimum probability for exploration
        probs = np.clip(probs, self.min_prob, 1.0)
        probs = probs / probs.sum()  # Renormalize
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

    def learn(self, gamma=0.995):
        """
        Perform the policy update using the collected rewards and actions over an episode.

        Parameters:
        - gamma (float): Discount factor for future rewards (default 0.99).
        """
        G = 0
        returns = []
        
        # Compute the returns (discounted sum of future rewards) for each time step
        for r in reversed(self.episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        # Convert returns to a numpy array
        returns = np.array(returns)
        
        # Update reward statistics
        self.reward_count += len(returns)
        self.reward_mean = (self.reward_mean * (self.reward_count - len(returns)) + returns.sum()) / self.reward_count
        self.reward_std = np.sqrt(((self.reward_std ** 2) * (self.reward_count - len(returns)) + 
                                 ((returns - self.reward_mean) ** 2).sum()) / self.reward_count)
        
        # Normalize returns using running statistics
        returns = (returns - self.reward_mean) / (self.reward_std + 1e-8)

        # Apply learning rate decay
        self.lr *= self.lr_decay
        self.lr = max(self.lr, 1e-4)

        # Update the policy for each state-action pair
        for state, action, Gt in zip(self.episode_states, self.episode_actions, returns):
            baseline = self.policy[state]
            grad = np.eye(self.n_actions)[action] - baseline
            
            # Entropy regularization
            entropy_grad = -np.log(baseline + 1e-10) - 1
            total_grad = Gt * grad + self.entropy_coef * entropy_grad
            
            # Gradient clipping
            total_grad = np.clip(total_grad, -1.0, 1.0)
            
            # Update the policy
            self.policy[state] += self.lr * total_grad
            
            # Ensure policy probabilities are valid and maintain minimum probability
            self.policy[state] = np.clip(self.policy[state], self.min_prob, 1)
            self.policy[state] /= self.policy[state].sum()

        # Reset episode data after learning
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Apply entropy coefficient decay
        self.entropy_coef *= 0.995
        self.entropy_coef = max(self.entropy_coef, 0.001)

        # Apply minimum probability decay
        self.min_prob *= 0.995
        self.min_prob = max(self.min_prob, 1e-4)
        
        # Apply temperature decay
        self.temperature *= 0.995
        self.temperature = max(self.temperature, 0.1)

    def get_greedy_action(self, state):
        probs = self.policy[state]
        return np.argmax(probs) 