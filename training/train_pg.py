from agent.policy_gradient_agent import PolicyGradientAgent
from visualization.learning_curve import plot_learning_curve

def train_pg_agent(env, num_episodes):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = PolicyGradientAgent(state_size, action_size)
    rewards = []
    steps = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        total_steps = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            total_steps += 1

        agent.learn()
        rewards.append(total_reward)
        steps.append(total_steps)
        plot_learning_curve(values=rewards, agent_type="Policy Gradient", curve_type="Rewards")
        #plot_learning_curve(values=steps, agent_type="Policy Gradient", curve_type="Steps")
        print(f"Episode {(episode+1):>3}/{num_episodes} - Total Reward: {total_reward:>4} - Total Steps: {total_steps:>4}")

    return agent

