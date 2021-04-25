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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, predictions, actions, discounted_rewards):
        action_probability = []
        for (action_, prediction_) in zip(actions, predictions):
            action_probability.append(action_ * prediction_)
        action_probability = torch.cat(action_probability)
        action_probability = torch.sum(action_probability, dim=1)
        cross_entropy = torch.log(action_probability) * discounted_rewards
        loss = - torch.sum(cross_entropy)
        return loss


class PGAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = list(range(5))
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.005

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        print('Using {} device'.format(self.device))
        self.model = PolicyNN(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.RMSprop(lr=self.learning_rate, params=self.model.parameters())
        self.loss_fn = PolicyLoss()

        self.states, self.actions, self.rewards, self.predictions = [], [], [], []

        if self.load_model:
            self.model.load_state_dict(torch.load('./save_model/reinforce_trained.pth'))

    def get_action(self, state_):
        prediction = self.model(state_)
        policy = prediction[0]
        # print(prediction, "\n", policy, "\n", type(prediction), type(policy))
        dist = torch.distributions.Categorical(policy)
        action_ = dist.sample()
        self.predictions.append(prediction)
        return action_.item()

    def discount_rewards(self, rewards_):
        discounted_rewards = torch.zeros_like(rewards_)
        running_add = 0.
        for t in reversed(range(len(rewards_))):
            running_add = running_add * self.discount_factor + rewards_[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state_, action_, reward_):
        self.states.append(state_[0])
        self.rewards.append(reward_)
        act = torch.zeros(self.action_size)
        act[action_] = 1.
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = self.discount_rewards(torch.tensor(self.rewards, dtype=torch.float32, device=self.device))
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        loss = self.loss_fn(
            self.predictions,
            self.actions,
            discounted_rewards)
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
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).view(1, 15)

        while not done:
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device).view(1, 15)

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
