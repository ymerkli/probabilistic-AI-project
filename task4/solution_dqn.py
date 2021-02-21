import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random

import time

import scipy.signal
from gym.spaces import Box, Discrete

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

SEED=7

def mlp(sizes, activation, output_activation=nn.Identity):
    """The basic multilayer perceptron architecture used."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class DQNet(nn.Module):
    """ Policy model """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, state):
        return self.logits_net(state)

class ReplayBuffer:
    def __init__(self, act_dim, size, batch_size):
        self.act_dim = act_dim
        self.memory = deque(maxlen=size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "act", "rew", "next", "done"])

    def add(self, state, act, rew, next, done):
        """Add a new experience to memory."""
        data = self.experience(state, act, rew, next, done)
        self.memory.append(data)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.act for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.rew for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQNAgent:
    def __init__(self, steps_per_epoch):
        # The observations are 8 dimensional vectors, and the actions are numbers,
        self.obs_dim = 8
        self.act_dim = 4

        self.lr = 1e-3
        self.batch_size = 64
        self.update_interval = 4

        # Discount factor for weighting future rewards
        self.gamma = 0.99
        self.lam = 0.97
        self.tau = 1e-3

        # Set up buffer
        self.buf = ReplayBuffer(self.act_dim, steps_per_epoch, self.batch_size)

        # initialize DQN networks
        self.hid = [64,64,64]  # layer width of networks
        self.qnetwork_local = DQNet(self.obs_dim, self.act_dim, self.hid, nn.ReLU)
        self.qnetwork_target = DQNet(self.obs_dim, self.act_dim, self.hid, nn.ReLU)

        self.optim = Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.t = 0

    def step(self, state, act, rew, next, done):
        self.buf.add(state, act, rew, next, done)
        self.t = (self.t + 1) % self.update_interval
        if self.t == 0:
            if len(self.buf) > self.batch_size:
                data = self.buf.sample()
                self.learn(data, self.gamma)

    def act(self, obs, eps=0):
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_logits = self.qnetwork_local(obs)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_logits.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.act_dim))

    def learn(self, data, gamma=0.99, tau=1e-3):
        """
        Arguments:
        experiences: Dictionary of torch variables (obs, act, rew, next, done)
        gamma: discount factor
        """
        state, act, rew, next, done = data
        q_targets_next = self.qnetwork_target(next).detach().max(1)[0]
        q_targets      = rew + (gamma * q_targets_next * (1 - done))
        q_expected     = self.qnetwork_local(state).gather(1, act)

        # minimize loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class Agent:
    def __init__(self, env):
        self.env = env

        # Training parameters
        self.steps_per_epoch = 1000
        self.epochs = 2000 # Number of epochs to train for
        self.dqna = DQNAgent(self.steps_per_epoch)

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.9

        self.print_interval = 100

    def train(self):
        """
        Main training loop.
        """
        # Main training loop: collect experience in env and update / log each epoch
        eps = self.eps_start
        scores = []
        scores_window = deque(maxlen=100)
        for epoch in range(self.epochs):
            state = self.env.reset()
            score = 0
            for t in range(self.steps_per_epoch):
                action = self.dqna.act(torch.as_tensor(state, dtype=torch.float32), eps)
                next_state, reward, terminal = self.env.transition(action)

                self.dqna.step(state, action, reward, next_state, terminal)
                # Update state (critical!)
                state = next_state
                score += reward

                if terminal:
                    break

            scores.append(score)
            scores_window.append(score)
            eps = max(self.eps_end, self.eps_decay*eps) # decrease epsilon

            if epoch % self.print_interval == 0:
                print(f'Epoch {epoch}\tAvg score: {np.mean(scores_window)}')

        return True

    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """
        return self.dqna.act(torch.Tensor(obs))


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()

    agent = Agent(env)
    agent.train()

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()
