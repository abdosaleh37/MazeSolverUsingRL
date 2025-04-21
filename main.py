import numpy as np
import time
import pickle
import os
from env.maze_env import MazeEnv
from agent.q_learning_agent import QLearningAgent

# إعداد البيئة
size = (10, 10)
env = MazeEnv(size=size)
agent = QLearningAgent(action_space=env.action_space)

# تدريب العميل
agent.q_table = np.zeros((env.observation_space.n, env.action_space.n))
num_episodes = 2000
q_table_path = "training/q_table.pkl"

# بداية التدريب إذا لم يكن هناك Q-table موجود
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

    # حفظ الـ Q-table المدرب
    with open(q_table_path, "wb") as f:
        pickle.dump(agent.q_table, f)

else:
    # تحميل Q-table إذا كان موجود
    with open(q_table_path, "rb") as f:
        agent.q_table = pickle.load(f)

# عرض حركة عشوائية قبل التعلم
print("Maze before learning:")
# state = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # حركة عشوائية
#     next_state, reward, done, _ = env.step(action)
#     state = next_state
#     env.render()
#     time.sleep(0.01)

# عرض أفضل مسار بعد التعلم
print("Best path after learning:")
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)  # استخدام الـ Q-table لاختيار الحركة
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.1)
