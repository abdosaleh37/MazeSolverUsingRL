import numpy as np

class QLearningAgent:
    def __init__(self, action_space, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.action_space = action_space
        self.epsilon = epsilon  # درجة الاستكشاف
        self.alpha = alpha  # معدل التعلم
        self.gamma = gamma  # عامل الخصم
        self.q_table = {}  # جدول Q لتخزين القيم

    def get_action(self, state):
        if np.random.rand() < self.epsilon:  # استكشاف
            return np.random.choice(self.action_space.n)  # اختر إجراء عشوائيًا
        else:  # استغلال
            return np.argmax(self.q_table.get(state, np.zeros(self.action_space.n)))  # اختر أفضل إجراء بناءً على Q-table

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table.get(state, np.zeros(self.action_space.n))[action]
        max_future_q = np.max(self.q_table.get(next_state, np.zeros(self.action_space.n)))
        # تحديث القيمة في Q-table
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table.setdefault(state, np.zeros(self.action_space.n))[action] = new_q
