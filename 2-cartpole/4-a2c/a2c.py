import sys
import gym
import pylab
import numpy as np

import torch
import torch.nn as nn

EPISODES = 1000


class ActorNN(nn.Module):
    def __init__(self, input_dim, output_dim):

        def init_linear(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)

        super(ActorNN, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim),
            nn.Softmax(dim=1)
        )

        self.NN.apply(init_linear)

    def forward(self, x):
        return self.NN(x)


class CriticNN(nn.Module):
    def __init__(self, input_dim, output_dim):

        def init_linear(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)

        super(CriticNN, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )

        self.NN.apply(init_linear)

    def forward(self, x):
        return self.NN(x)


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.discount_factor = 0.99
        self.lr_actor = 0.001
        self.lr_critic = 0.005

        self.actor_nn = ActorNN(self.state_size, self.action_size)
        self.critic_nn = CriticNN(self.state_size, self.value_size)
        self.optimizer_actor = torch.optim.Adam(lr=self.lr_actor, params=self.actor_nn.parameters())
        self.optimizer_critic = torch.optim.Adam(lr=self.lr_critic, params=self.critic_nn.parameters())
        self.loss_actor = nn.CrossEntropyLoss()
        self.loss_critic = nn.MSELoss()

        if self.load_model:
            self.actor_nn.load_state_dict(torch.load("./save_model/cartpole_actor.pth"))
            self.critic_nn.load_state_dict(torch.load("./save_model/cartpole_critic.pth"))

    def get_action(self, state_):
        prediction = self.actor_nn(state_)
        dist = torch.distributions.Categorical(prediction)
        return dist.sample().item()

    def train_model(self, state_, action_, reward_, next_state_, done_):
        prediction_actor = self.actor_nn(state_)
        prediction_critic = self.critic_nn(state_)
        prediction_critic_next = self.critic_nn(next_state_)

        value = prediction_critic[0].detach()
        value_next = prediction_critic_next[0].detach()

        advantages = torch.zeros(1, self.action_size, requires_grad=False)
        target = torch.zeros(1, self.value_size, requires_grad=False)

        if done_:
            advantages[0][action_] = reward_ - value
            target[0][0] = reward_
        else:
            advantages[0][action_] = reward_ + self.discount_factor * value_next - value
            target[0][0] = reward_ + self.discount_factor * value_next
        loss_actor = torch.sum(-torch.log(prediction_actor) * advantages, dim=1).mean()
        # Here, cross entropy loss with reward maybe a little bit confusing
        # I am just trying not to make too many changes to the original code
        # The one hot building process is retained, which already placed the reward into the
        # advantage variable. So, no need for another multiplication or discount reward
        loss_critic = self.loss_critic(prediction_critic, target)
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).view(1, state_size)

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1, state_size)
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            torch.save(agent.actor_nn.state_dict(), "./save_model/cartpole_actor.pth")
            torch.save(agent.critic_nn.state_dict(), "./save_model/cartpole_critic.pth")