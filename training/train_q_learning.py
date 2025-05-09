from agent.q_learning_agent import QLearningAgent
from training.train_pg import plot_rewards

def train_q_agent(env, num_episodes):
    state_size = env.observation_space.n
    action_space = env.action_space.n
    agent = QLearningAgent(action_space=action_space, state_size=state_size)
    rewards = []
        
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        plot_rewards(rewards, agent_type="Q-Learning")
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")
    
    q_table_path="training/q_table.pkl"
    agent.save(q_table_path)
    print("Training complete and Q-table saved.")
    
    return agent




    
