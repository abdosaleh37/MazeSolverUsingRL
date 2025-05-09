from matplotlib import pyplot as plt

def plot_learning_curve(values, agent_type, curve_type):
    """
    Plots the learning curve of the agent during training.

    Parameters:
    - values: A list of values representing either rewards or steps accumulated during training.
    - agent_type: The type of agent used (e.g., "Policy Gradient" or "Q Learning").
    - curve_type: The type of curve to plot, either "Rewards" or "Steps".

    This function plots the graph of values against episodes and updates the plot dynamically.
    """
    # Create a new figure for the curve, with a title based on the curve type 
    plt.figure(f"{curve_type} Curve")
    
    # Clear the current figure to prepare for new plot data
    plt.clf()
    
    # Plot the values against the episodes (values list)
    plt.plot(values)
    
    # Label the x-axis as "Episode"
    plt.xlabel("Episode")
    
    # Label the y-axis dynamically based on the curve type
    plt.ylabel(f"Total {curve_type}")
    
    # Set the title of the plot to reflect the agent type and curve type
    plt.title(f"{agent_type}: {curve_type} Curve")
    
    # Enable grid lines for better visualization of the curve
    plt.grid(True)
    plt.draw()
    plt.pause(0.05)