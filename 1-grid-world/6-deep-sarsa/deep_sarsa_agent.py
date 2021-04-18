import copy
import pylab
import random
import numpy as np

from environment import Env
from torch import nn
import torch

EPISODES = 1000


class SarsaNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SarsaNN, self).__init__()
        self.dense_stack = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, output_dim),
        )

    def forward(self, x):
        return self.dense_stack(x)


class DeepSARSAAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = list(range(5))
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        self.model = SarsaNN(self.state_size, self.action_size).to(self.device)
        # self.model = SarsaNN(self.state_size, self.action_size)
        print("Model structure: ", self.model, "\n\n")
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_state_dict(torch.load("./save_model/deep_sarsa.pth"))

    def get_action(self, state_):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_ = torch.tensor(state_, device=self.device, dtype=torch.float32)
            q_values = self.model(state_)
            return torch.argmax(q_values[0])

    def train_model(self, state_, action_, reward_, next_state_, next_action_, done_):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state_ = torch.tensor(state_, device=self.device, dtype=torch.float32)
        next_state_ = torch.tensor(next_state_, device=self.device, dtype=torch.float32)
        pred = self.model(state_)
        target = pred[0]

        if done_:
            target[action_] = reward_
        else:
            target[action_] = reward_ + self.discount_factor * self.model(next_state_)[0][next_action_]

        target = target.view(1, 5)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = Env()
    agent = DeepSARSAAgent()
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_sarsa_agent.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        if e % 100 == 0:
            torch.save(agent.model.state_dict(), "./save_model/deep_sarsa.pth")
