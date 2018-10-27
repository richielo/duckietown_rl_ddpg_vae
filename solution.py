#!/usr/bin/env python
import traceback

import gym
import numpy as np

# noinspection PyUnresolvedReferences
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
from duckietown_challenges import wrap_solution, ChallengeSolution, ChallengeInterfaceSolution, InvalidEnvironment
from wrappers import SteeringToWheelVelWrapper


def solve(params, cis):
    # python has dynamic typing, the line below can help IDEs with autocompletion
    assert isinstance(cis, ChallengeInterfaceSolution)
    # after this cis. will provide you with some autocompletion in some IDEs (e.g.: pycharm)
    cis.info('Creating model.')
    # you can have logging capabilties through the solution interface (cis).
    # the info you log can be retrieved from your submission files.

    # We get environment from the Evaluation Engine
    cis.info('Making environment')
    env = gym.make(params['env'])

    # === BEGIN SUBMISSION ===

    # If you created custom wrappers, you also need to copy them into this folder.

    from wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper

    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    # to make the images pytorch-conv-compatible
    env = ImgWrapper(env)
    env = ActionWrapper(env)

    # you ONLY need this wrapper if you trained your policy on [speed,steering angle]
    # instead [left speed, right speed]
    env = SteeringToWheelVelWrapper(env)

    # you have to make sure that you're wrapping at least the actions
    # and observations in the same as during training so that your model
    # receives the same kind of input, because that's what it's trained for
    # (for example if your model is trained on grayscale images and here
    # you _don't_ make it grayscale too, then your model wont work)

    # HERE YOU NEED TO CREATE THE POLICY NETWORK SAME AS YOU DID IN THE TRAINING CODE
    # if you aren't using the DDPG baseline code, then make sure to copy your model
    # into the model.py file and that it has a model.predict(state) method.
    from model import DDPG

    model = DDPG(state_dim=env.observation_space.shape, action_dim=2, max_action=1, net_type="cnn")

    try:
        model.load("model", "models")

        # === END SUBMISSION ===

        # Then we make sure we have a connection with the environment and it is ready to go
        cis.info('Reset environment')
        observation = env.reset()

        # While there are no signal of completion (simulation done)
        # we run the predictions for a number of episodes, don't worry, we have the control on this part
        while True:
            # we passe the observation to our model, and we get an action in return
            action = model.predict(observation)
            # we tell the environment to perform this action and we get some info back in OpenAI Gym style
            observation, reward, done, info = env.step(action)
            # here you may want to compute some stats, like how much reward are you getting
            # notice, this reward may no be associated with the challenge score.

            # it is important to check for this flag, the Evalution Engine will let us know when should we finish
            # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
            # from this submission
            if 'simulation_done' in info:
                cis.info('simulation_done received.')
                break
            if done:
                cis.info('Episode done; calling reset()')
                env.reset()

    finally:
        # release CPU/GPU resources, let's be friendly with other users that may need them
        cis.info('Releasing resources')
        try:
            model.close()
        except:
            msg = 'Could not call model.close():\n%s' % traceback.format_exc()
            cis.error(msg)
    cis.info('Graceful exit of solve()')



class Submission(ChallengeSolution):
    def run(self, cis):
        assert isinstance(cis, ChallengeInterfaceSolution)  # this is a hack that would help with autocompletion

        # get the configuration parameters for this challenge
        params = cis.get_challenge_parameters()
        cis.info('Parameters: %s' % params)

        cis.info('Starting.')
        solve(params, cis)

        cis.set_solution_output_dict({})
        cis.info('Finished.')


if __name__ == '__main__':
    print('Starting submission')
    wrap_solution(Submission())
