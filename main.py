import time
from env.maze_env import MazeEnv
from training.train_q_learning import train_q_agent
from training.train_pg import train_pg_agent
from visualization.select_agent import select_agent_window

def main():
    # Display the agent selection window and store the selected agent in a dictionary
    selected_agent = select_agent_window()
    
    # Check if an agent was selected. If not, print a message and exit
    if selected_agent['type'] is None:
        print('No agent selected. Exiting.')
        return

    # Set the number of episodes for training and the size of the maze
    num_episodes = 200
    size = (15, 15)
    
    # Initialize the maze environment with the specified size
    env = MazeEnv(size=size)

    # Train the agent based on the selection
    if selected_agent['type'] == 'q':
        agent = train_q_agent(env=env, num_episodes=num_episodes)
    else:
        agent = train_pg_agent(env=env, num_episodes=num_episodes)

    print("\nTraining complete! Starting visualization of the learned path...")
    print("Press Enter to close the visualization window when done.")
    
    # Start the simulation after training
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()  # This will now create the Pygame window only when first called
        state = next_state
        time.sleep(0.1)
    
    # Wait for the user to press Enter before closing the program
    input('Press Enter to close...')
    env.close()

if __name__ == '__main__':
    main()