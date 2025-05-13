from agent.q_learning_agent import QLearningAgent
from training.train_pg import plot_learning_curve

def train_q_agent(env, num_episodes):
    """
    Train a Q-learning agent on a given environment.

    Parameters:
    - env: The environment in which the agent will be trained.
    - num_episodes: The number of episodes to run during training.

    Returns:
    - agent: The trained Q-learning agent.
    """
    # Initialize the state size and action space based on the environment's spaces
    state_size = env.observation_space.n
    action_space = env.action_space.n
    
    # Instantiate the QLearningAgent with the environment's parameters
    agent = QLearningAgent(action_space=action_space, state_size=state_size)
    
    # Lists to track rewards and steps for each episode
    rewards = []
    steps = []
    
    # Main training loop for the given number of episodes
    for episode in range(num_episodes):
        # Reset the environment at the start of each episode
        state = env.reset()
        done = False
        total_reward = 0
        total_steps = 0
        
        # Step through the environment until the episode ends
        while not done:
            # Get the action to take based on the current state (using epsilon-greedy strategy)
            action = agent.get_action(state)
            
            # Perform the action in the environment and observe the result
            next_state, reward, done, _ = env.step(action)
            
            # The agent learns from the transition (state, action, reward, next_state)
            agent.learn(state, action, reward, next_state, done)
            
            # Update the state for the next iteration
            state = next_state
            total_reward += reward
            total_steps += 1
        
        # After the episode is done, append the results (total reward and total steps)
        rewards.append(total_reward)
        steps.append(total_steps)
    
        # Print the result of the current episode
        print(f"Episode {(episode+1):>3}/{num_episodes} - Total Reward: {total_reward:>4} - Total Steps: {total_steps:>4}")
    
    # Visualize both rewards and steps in one plot
    plot_learning_curve(rewards=rewards, steps=steps, agent_type="Q Learning")
    
    # Return the trained agent
    return agent




    
