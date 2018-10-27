# AI Driving Olympics - PyTorch/DDPG Baseline

<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/07/AIDO-768x512.png"/></a>


## Warning

In the scripts "2-/3-" currently predict the action in the format `[speed,steering angle]` and not the format that is required for this challenge, `[speed left wheel, speed right wheel]`. This can be fixed by using the scripts `4-/5-` or by waiting for a bit until we add the inverse kinematics wrapper.

## Quickstart

To train a policy,... 

(We assume, you have Python3 and docker installed)

0) Install Hyperdash (an open source experiment tracker and plotting tool), and create a profile or login (for more info check out [hyperdash.io](hyperdash.io)

        pip install hyperdash && hyperdash signup # for conda users
        # OR for system-wide installation
        sudo pip install hyperdash && hyperdash signup
    
1) Clone this repo

        git clone https://github.com/duckietown/challenge-aido1_LF1-baseline-RL-sim-pytorch.git

2) Change into the directory that you cloned:
    
        cd challenge-aido1_LF1-baseline-pytorch
        
3) Install this package

        pip3 install -e . # if you are in a conda env
        # or if you want to install this system-wide
        sudo pip3 install -e . 
        
4) Now change into the scripts directory

        cd scripts
        
5) Now you have two options. You can train on environment **(a)** that is **local**, randomized - therefore easier to transfer to the real robot -, more customizable, easier to debug, and faster but it might not run on your machine and there is option **(b)** the **docker-based** environment that is the opposite but runs everywhere.

---

- (5a) Run the training script

        python 2-train-ddpg-cnn.py --seed 123
        
- (6a) When it finishes, check it out (but first edit this following file and set the seed to the one you used above, like `123` in line 10)

        python 3-test-ddpg-cnn.py
        
---
        
- (5b) Start the gym-duckietown-server and keep it running in the background for both training and testing 

        docker run -tid -p 8902:8902 -p 5558:5558 \
        -e DISPLAY=$DISPLAY -e DUCKIETOWN_CHALLENGE=LF \
        --name gym-duckietown-server --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        duckietown/gym-duckietown-server

- (6b) Run the training script

        python 4-train-ddpg-cnn-remote.py --seed 123
        
- (7b) When it finishes, check it out (but first edit this following file and set the seed to the one you used above, like `123` in line 10)

        python 5-test-ddpg-cnn-remote.py
        
- (8b) When you're done with the dockering, first run
 
        docker ps
        
        # copy the "CONTAINER ID" of the running container
        
        docker container stop INSERT_ID_HERE

## Template "PyTorch template" for challenge `aido1_LF1-v3`

This is baseline to help you train a driving policy for the challenges in the [the AI Driving Olympics](http://aido.duckietown.org/).

The [online description of this challenge is here][online].

For submitting, please follow [the instructions available in the book][book].
 
[book]: http://docs.duckietown.org/DT18/AIDO/out/

[online]: https://challenges.duckietown.org/v3/humans/challenges/aido1_LF1-v3

## Description

This is a simple template for an agent that uses PyTorch/DDPG for inference.

[This code is documented here](https://docs.duckietown.org/DT18/AIDO/out/pytorch_baseline.html).

## How to submit the trained policy

Once you're done training, you need to copy your model and the saved weights of the policy network.

Specifically if you use this repo then you need to copy over

- `duckietown_rl/ddpg.py` and rename to `model.py`
- `scripts/pytorch_models/DDPG_2_XXX_actor.pth` and `..._XXX_critic.pth` and rename to `models/model_actor.pth` and `models/model_critic.pth` respectively, where `XXX` is the seed of your best policy
- `duckietown_rl/wrappers.py` to just `wrappers.py` and don't rename. :)

And then edit the `solution.py` file over to make sure everything is loaded up correctly (i.e all of the imports point to the right place) and execute `dts challenges submit` to send your solution.

## How to improve your policy

Here are some idea for improving your policy:

- Check out the `DtRewardWrapper` and modify the rewards (set them higher or lower and see what happens)
- Try making the observation image grayscale instead of color. And while you're at it, try stacking multiple images, like 4 monochrome images instead of 1 color image
- You can also try increasing the contrast in the input. to make the difference between road and road-signs clearer. You can do so by adding another observation wrapper.
- Cut off the horizon from the image (and correspondingly change the convnet parameters). 
- Check out the default hyperparameters in `duckietown_rl/args.py` and fiddle around with them; see if they work better. For example increase the `expl_noise` or increase the `start_timesteps` to get better exploration.
- (more sophisticated) Use a different map in the simulator, or - even better - use randomized maps.
- (more sophisticated) Use a different/bigger convnet for your actor/critic. And add better initialization.
- (super sophistacted) Use the ground truth from the simulator to construct a better reward  
- (crazy sophisticated) Use an entirely different training algorithm - like PPO, A2C, or DQN. Go nuts. But this might take significant time, even if you're familiar with the matter.

## How to track experiment progress

Install the HyperDash app for Android/iOS. You can follow the progress of he various experiments there and you'll get a push notification when your experiment finishes or breaks for some reason.
