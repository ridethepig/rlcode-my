import sys
import gym
import pylab
import numpy as np

import torch
import torch.nn as nn

EPISODES = 1000


class PGNN(nn.Module):
    def __init__(self, input_dim, output_dim):

        def init_linear(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)

        super(PGNN, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim),
            nn.Softmax(dim=1)
        )
        self.NN.apply(init_linear)

    def forward(self, x):
        return self.NN(x)


class PolicyLoss(nn.Module):
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, predictions, actions, discounted_rewards):
        action_probability = []
        for (action_, prediction_) in zip(actions, predictions):
            action_probability.append(action_ * prediction_)
        action_probability = torch.cat(action_probability)
        action_probability = torch.sum(action_probability, dim=1)
        loss = - torch.log(action_probability) * discounted_rewards
        loss = loss.sum()
        return loss


class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.pg_net = PGNN(state_size, action_size)
        self.optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.pg_net.parameters())
        self.loss_fn = PolicyLoss()

        self.states = []
        self.actions = []
        self.rewards = []
        self.predictions = []

    def get_action(self, state_):
        policy = self.pg_net(state_)
        dist = torch.distributions.Categorical(policy)
        self.predictions.append(policy)
        return dist.sample().item()

    def discount_reward(self, rewards_):
        discounted_rewards = torch.zeros(len(rewards_), dtype=torch.float32)
        running_add = 0.
        for t in reversed(range(len(rewards_))):
            running_add = running_add * self.discount_factor + rewards_[t]
            discounted_rewards[t] = running_add
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        return discounted_rewards

    def append_sample(self, state_, action_, reward_):
        self.states.append(state_)
        act = torch.zeros(self.action_size, 1)
        act[action_] = 1.
        self.actions.append(act.T)
        self.rewards.append(reward_)

    def train_model(self):
        discounted_rewards = self.discount_reward(self.rewards)

        loss = self.loss_fn(
            self.predictions,
            self.actions,
            discounted_rewards
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.predictions = []


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).view(1, state_size)

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1, state_size)
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_reinforce.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            torch.save(agent.pg_net.state_dict(), "./save_model/cartpole_reinforce.pth")