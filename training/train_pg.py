from agent.policy_gradient_agent import PolicyGradientAgent
from visualization.learning_curve import plot_learning_curve

def train_pg_agent(env, num_episodes):
    """
    Train a Policy Gradient agent on a given environment.

    Parameters:
    - env: The environment in which the agent will be trained.
    - num_episodes: The number of episodes to run during training.

    Returns:
    - agent: The trained Policy Gradient agent.
    """
    # Initialize the state and action size based on the environment's spaces
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Instantiate the PolicyGradientAgent
    agent = PolicyGradientAgent(state_size, action_size)
    
    # Lists to track rewards and steps for each episode
    rewards = []
    steps = []

    # Main training loop for the given number of episodes
    for episode in range(num_episodes):
        # Reset the environment at the start of each episode
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        total_steps = 0
        
        # Step through the environment until the episode ends
        while not done:
            # Get the action to take based on the current state
            action = agent.get_action(state)
            
            # Perform the action in the environment and observe the result
            next_state, reward, done, _ = env.step(action)
            
            # Store the reward for the current step (used later in learning)
            agent.store_reward(reward)
            
            # Update the state for the next iteration
            state = next_state
            
            total_reward += reward
            total_steps += 1

        agent.learn()
        rewards.append(total_reward)
        steps.append(total_steps)
        
        # Print the result of the current episode
        print(f"Episode {(episode+1):>3}/{num_episodes} - Total Reward: {total_reward:>4} - Total Steps: {total_steps:>4}")

    # Visualize both rewards and steps in one plot
    plot_learning_curve(rewards=rewards[3:], steps=steps[3:], agent_type="Policy Gradient")
    
    # Return the trained agent
    return agent

