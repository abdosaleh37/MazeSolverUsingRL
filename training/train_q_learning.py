from agent.q_learning_agent import QLearningAgent
import pickle
import matplotlib.pyplot as plt


def train_q_agent(env , num_episodes=2000, q_table_path="training/q_table.pkl"):
    state_size = env.observation_space.n
    agent = QLearningAgent(action_space=env.action_space, state_size=state_size)
    rewards = []
    
    plt.ion()
    plt.figure("Learning Curve")
    
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
        draw_learning_curve(rewards, episode, num_episodes, total_reward)

    plt.ioff()
    plt.show(block=False)
    
    with open(q_table_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Training complete and Q-table saved.")
    
    return agent


def draw_learning_curve(rewards, episode, num_episodes, total_reward):
    plt.figure("Learning Curve")
    plt.clf()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning: Learning Curve")
    plt.grid(True)
    plt.draw()
    plt.pause(0.05)
    print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")

    
