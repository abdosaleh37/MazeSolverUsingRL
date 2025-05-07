import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradientAgent:
    def __init__(self, input_size, output_size):
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)
    
    def get_action(self, state):
        state_array = np.array(state)
        state_tensor = torch.tensor(state_array.flatten(), dtype=torch.float32)

        probs = self.forward(state_tensor)
        action = torch.multinomial(probs, 1)
        return action.item() 

    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])

        probs = self(state_tensor)
        prob_distribution = torch.nn.functional.softmax(probs, dim=-1)
        action_prob = prob_distribution[0, action_tensor]

        loss = -torch.log(action_prob) * reward_tensor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
