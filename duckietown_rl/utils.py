import random

import gym
import numpy as np
import torch


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        self.storage.append((state, next_state, action, reward, done))

    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    return avg_reward
