# training/train_q_learning.py
from env.maze_env import MazeEnv
from agent.q_learning_agent import QLearningAgent
import pickle

def train_q_learning(size=(10, 10), num_episodes=2000, q_table_path="training/q_table.pkl"):
    env = MazeEnv(size=size)
    agent = QLearningAgent(action_space=env.action_space)

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

        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")

    with open(q_table_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Training complete and Q-table saved.")
