import time
from env.maze_env import MazeEnv
from training.train_q_learning import train_q_agent
from training.train_pg import train_pg_agent
from visualization.select_agent import select_agent_window

def main():
    selected_agent = select_agent_window()
    if selected_agent['type'] is None:
        print('No agent selected. Exiting.')
        return

    num_episodes = 300
    size = (10, 10)
    env = MazeEnv(size=size)

    if selected_agent['type'] == 'q':
        agent = train_q_agent(env=env, num_episodes=num_episodes)
    else:
        agent = train_pg_agent(env=env, num_episodes=num_episodes)

    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        time.sleep(0.1)
    input('Press Enter to close...')

if __name__ == '__main__':
    main()