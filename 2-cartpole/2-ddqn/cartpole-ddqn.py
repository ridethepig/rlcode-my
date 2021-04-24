import gym
import pylab
import random
import sys
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 400
dtype = torch.float32


class QNN(nn.Module):
    # in this example, we map state directly to action
    def __init__(self, input_dim, output_dim):
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)
        super(QNN, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )
        self.NN.apply(init_normal)

    def forward(self, x):
        return self.NN(x)


class DQNAgent:
    def __init__(self, state_size_, action_size_):
        self.render = False
        self.load_model = False

        self.state_size = state_size_
        self.action_size = action_size_

        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.epsilon = 0.95
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000

        self.memory = deque(maxlen=2000)

        # We use Fixed Q-target method to avoid
        # using the same network to predict our Q-value and Q-target
        # which causes huge fluctuation and takes long to converge
        # self.device = 'cuda'
        self.device = 'cpu'  # in this case, cpu is much much faster
        self.policy_net = QNN(state_size_, action_size_).to(self.device)
        self.target_net = QNN(state_size_, action_size_).to(self.device)
        self.optimizer = optim.Adam(lr=self.learning_rate, params=self.policy_net.parameters())
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

        if self.load_model:
            self.policy_net.load_state_dict(torch.load("./save_model/cartpole_ddqn.pth"))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state_):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value_ = self.policy_net(state_)
            return torch.argmax(q_value_[0]).item()

    def append_sample(self, state_, action_, reward_, next_state_, done_):
        self.memory.append((state_, action_, reward_, next_state_, done_))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = torch.zeros(self.batch_size, self.state_size, dtype=dtype, device=self.device)
        update_target = torch.zeros(self.batch_size, self.state_size, dtype=dtype, device=self.device)
        action_, reward_, done_ = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action_.append(mini_batch[i][1])
            reward_.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done_.append(mini_batch[i][4])

        q_target = self.policy_net(update_input)
        q_next_target = self.policy_net(update_target).detach()
        q_value = self.target_net(update_target).detach()
        q_target_origin = q_target.clone()
        '''
        Only about two lines of different code from Fixed Q-target
            we use our policy net to predict the next action instead of 
            choosing the maximum produced by target net.
            It seems to be quite reasonable, 'cause the policy net is much
            fresher than the target net, which has greater choices to produce
            better predictions 
        '''

        for i in range(self.batch_size):
            if done_[i]:
                q_target[i][action_[i]] = reward_[i]
            else:
                a = torch.argmax(q_next_target[i])
                q_target[i][action_[i]] = reward_[i] + self.discount_factor * q_value[i][a]

        loss = self.loss_fn(q_target_origin, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    time_start = datetime.now()

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = torch.tensor(state, dtype=dtype, device=agent.device).view(1, state_size)

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=dtype, device=agent.device).view(1, state_size)
            reward = reward if not done or score == 499 else -100
            agent.append_sample(state, action, reward, next_state, done)

            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_network()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_ddqn.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
        if e % 50 == 0:
            torch.save(agent.policy_net.state_dict(), "./save_model/cartpole_ddqn.pth")
            print(datetime.now() - time_start)
            time_start = datetime.now()
