import time
from env.maze_env import MazeEnv
from training.train_q_learning import train_q_agent

size = (10, 10)
num_episodes = 300
env = MazeEnv(size=size)

agent = train_q_agent(env=env, num_episodes=num_episodes)

state = env.reset()
done = False
while not done:
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.1)
    
input("Press Enter to close...")
