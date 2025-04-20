from agent import QLearningAgent, PolicyGradientAgent
from env.maze_env import MazeEnv
import time


env = MazeEnv(size=(20, 20))  # Size of the maze
# agent_type = input("Select agent (q or pg): ").strip()

# if agent_type == "q":
agent = QLearningAgent(action_space=env.action_space)
# elif agent_type == "pg":
#     agent = PolicyGradientAgent(input_size=env.observation_space.n, output_size=env.action_space.n)
# else:
#     raise ValueError("Wrong Choice!")



# Training
num_episodes = 2000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state) 
        next_state, reward, done, _ = env.step(action)  
        total_reward += reward

        # Update based on result
        agent.learn(state, action, reward, next_state, done)
        state = next_state

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

print("\n** Training Complete. Now displaying the solution path. **\n")

# Best path found
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.1)
