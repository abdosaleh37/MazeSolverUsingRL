from matplotlib import pyplot as plt

def plot_learning_curve(values, agent_type, curve_type):
    plt.figure(f"{curve_type} Curve")
    plt.clf()
    plt.plot(values)
    plt.xlabel("Episode")
    plt.ylabel(f"Total {curve_type}")
    plt.title(f"{agent_type}: {curve_type} Curve")
    plt.grid(True)
    plt.draw()
    plt.pause(0.05)