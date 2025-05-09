from matplotlib import pyplot as plt

# Create a global figure and axes that will be reused
fig = None
ax1 = None
ax2 = None

def plot_learning_curve(rewards, steps, agent_type):
    """
    Plots rewards and steps curves in separate subplots side by side.

    Parameters:
    - rewards: A list of reward values accumulated during training.
    - steps: A list of step counts accumulated during training.
    - agent_type: The type of agent used (e.g., "Policy Gradient" or "Q Learning").
    """
    global fig, ax1, ax2
    
    # Create figure and subplots only if they don't exist
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f'{agent_type}: Learning Curves', fontsize=12)
        plt.tight_layout()
    
    # Clear the previous plots
    ax1.clear()
    ax2.clear()
    
    # Plot rewards in the left subplot
    ax1.plot(rewards, color='tab:blue', label='Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot steps in the right subplot
    ax2.plot(steps, color='tab:red', label='Steps')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Update the plot
    plt.draw()
    plt.pause(0.05)