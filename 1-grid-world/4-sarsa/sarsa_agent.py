import numpy as np
import random
from collections import defaultdict
from environment import Env


class SARSAAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.007
        self.discount_factor = 0.9
        self.epsilon = 0.25
        self.q_table = defaultdict(lambda: [0., 0., 0., 0.])
        self.e_table = defaultdict(lambda: [0., 0., 0., 0.])
        self.lmbd = 0.9

    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q)
        self.q_table[state][action] = new_q

    def learn_lambda(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]

        delta = reward + self.discount_factor * next_state_q - current_q
        self.e_table[state][action] += 1.
        for s in self.e_table:
            for a, _ in enumerate(self.e_table[s]):
                self.q_table[s][a] += self.learning_rate * delta * self.e_table[s][a]
                self.e_table[s][a] *= self.discount_factor * self.lmbd

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    # To provide a relatively more interesting argmax
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()
    agent = SARSAAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        if agent.epsilon >= 0.08:
            agent.epsilon *= 0.99
        step_count = 0
        state = env.reset()
        agent.e_table.clear()
        action = agent.get_action(str(state))
        if agent.epsilon > 0.05:
            agent.epsilon *= 0.99
        while True:
            env.render()
            next_state, reward, done = env.step(action)
            if state == next_state:
                reward -= 1
            next_action = agent.get_action(str(next_state))
            agent.learn_lambda(str(state), action, reward, str(next_state), next_action)
            state = next_state
            action = next_action

            env.print_value_all(agent.q_table)

            step_count += 1
            if done:
                print("episode %d step %d" % (episode, step_count))
                break
