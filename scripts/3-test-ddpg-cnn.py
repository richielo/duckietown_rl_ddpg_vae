import gym
import gym_duckietown
import torch
from duckietown_rl.ddpg import DDPG
from duckietown_rl.utils import evaluate_policy
from duckietown_rl.args import get_ddpg_args_test
from duckietown_rl.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
import numpy as np

args = get_ddpg_args_test()

experiment = args.experiment
seed = args.seed
policy_name = "DDPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_name = "{}_{}_{}".format(
    policy_name,
    experiment,
    seed
)

env = gym.make("Duckietown-loop_obstacles-v0")

# Wrappers
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
# env = DtRewardWrapper(env) # not during testing

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

policy.load(file_name, directory="./pytorch_models")

with torch.no_grad():
    while True:
        obs = env.reset()
        env.render()
        rewards = []
        while True:
            action = policy.predict(np.array(obs))
            obs, rew, done, misc = env.step(action)
            rewards.append(rew)
            env.render()
            if done:
                break
        print ("mean episode reward:",np.mean(rewards))
