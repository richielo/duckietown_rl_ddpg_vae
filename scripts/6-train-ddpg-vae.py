import random

import numpy as np
import torch
import gym
import gym_duckietown
import os

from hyperdash import Experiment

from duckietown_rl import utils
from duckietown_rl.env import launch_env
from duckietown_rl.args import get_ddpg_args_train
from duckietown_rl.ddpg import DDPG
from duckietown_rl.vae import VAE
from duckietown_rl.utils import seed, evaluate_policy
from duckietown_rl.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper


from duckietown_rl.noise import OrnsteinUhlenbeckActionNoise as OUNoise    

experiment = 2
policy_name = "DDPG"
exp = Experiment("[duckietown] - ddpg-vae")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_train()

file_name = "{}_{}_{}".format(
    policy_name,
    experiment,
    str(args.seed),
)

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Launch the env with our helper function
env = launch_env()

# Wrappers
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)


# Set seeds
seed(args.seed)

# Use prioritized experience replay or not
use_pr = True

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Use ONoise 
use_onoise = True
if(use_onoise):
    noise_sigma = 0.2
    noise = OUNoise(mu=np.zeros(action_dim), sigma=noise_sigma)

# Load VAE
image_dimensions = 3 * 160 * 120
feature_dimensions = 1000
encoding_dimensions = 40
vae = VAE(image_dimensions, feature_dimensions, encoding_dimensions, 'selu')

if(use_pr):
    replay_buffer = utils.PrioritizedReplayBuffer(args.replay_buffer_max_size)
else:
    replay_buffer = utils.ReplayBuffer(args.replay_buffer_max_size)

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, replay_buffer, net_type="vae", vae=vae)


# Evaluate untrained policy
evaluations= [evaluate_policy(env, policy)]

exp.metric("rewards", evaluations[0])

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
while total_timesteps < args.max_timesteps:

    if done:

        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))
            exp.metric("rewards", evaluations[-1])

            if args.save_models:
                policy.save(file_name, directory="./pytorch_models")
            np.savez("./results/{}.npz".format(file_name),evaluations)

        # Reset environment
        env_counter += 1
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < 20:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs))
        if args.expl_noise != 0:
            if(use_onoise):
                on_noise = policy.epsilon * noise()
                action = (action + on_noise).clip(env.action_space.low, env.action_space.high)
            else:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

    # Perform action
    new_obs, reward, done, _ = env.step(action)

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    # Compute the target Q value
    if(use_pr):
        target_Q = policy.critic_target(np.expand_dims(obs,0), policy.actor_target(np.expand_dims(new_obs, 0))).data
        target_Q = reward + (done_bool * args.discount * target_Q)
        # Get current Q estimate
        current_Q = policy.critic(np.expand_dims(obs,0), action).data
        error = torch.abs(target_Q.cpu() - current_Q.cpu()).numpy()
        policy.replay_buffer.add(error, (obs, new_obs, action, reward, done_bool))
    else:
        policy.replay_buffer.add(obs, new_obs, action, reward, done_bool)

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))
exp.metric("rewards", evaluations[-1])

if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./results/{}.npz".format(file_name),evaluations)

exp.end()
