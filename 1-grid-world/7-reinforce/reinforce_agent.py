import copy
import pylab
import numpy as np

from environment import Env

import torch.nn as nn
import torch
import torch.functional as F

EPISODES = 2500


class PolicyNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)


class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, x, actions, discounted_rewards):
        action_prob = torch.sum(actions * x, dim=1)
        cross_entropy = torch.log(action_prob) * discounted_rewards
        loss = - torch.sum(cross_entropy)
        return loss


class PGAgent:
    def __init__(self):
        self.load_model = True
        self.action_space = list(range(5))
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = PolicyNN(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters())
        self.loss_fn = PolicyLoss()

        self.states, self.actions, self.rewards, self.predictions = [], [], [], []

        if self.load_model:
            self.model.load_state_dict(torch.load('./save_model/reinforce_trained.pth'))

    def get_action(self, state_):
        policy = self.model(state_)[0]
        self.predictions.append(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards_):
        discounted_rewards = torch.zeros_like(rewards_)
        running_add = 0
        for t in reversed(range(len(rewards_))):
            running_add = running_add * self.discount_factor + rewards_[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state_, action_, reward_):
        self.states.append(state_[0])
        self.rewards.append(reward_)
        act = torch.zeros(self.action_size)
        act[action_] = 1
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = torch.tensor(self.discount_rewards(self.rewards), dtype=torch.float32)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        loss = self.loss_fn(self.predictions, self.actions, discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.actions = []
        self.states = []
        self.predictions = []
        self.rewards = []


if __name__ == "__main__":
    env = Env()
    agent = PGAgent()
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = torch.tensor(state).view(1, 15)

        while not done:
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state).view(1, 15)

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score, "  time_step:", global_step)

        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/policy_gradient.png")
            torch.save(agent.model.state_dict(), "./save_model/policy_gradient.pth")
