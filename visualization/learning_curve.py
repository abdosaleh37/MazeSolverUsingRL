from matplotlib import pyplot as plt

def plot_rewards(rewards, agent_type):
    plt.figure("Learning Curve")
    plt.clf()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{agent_type}: Learning Curve")
    plt.grid(True)
    plt.draw()
    plt.pause(0.05)