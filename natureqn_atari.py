import argparse
import pdb

import gym
from utils.preprocess import greyscale, blackandwhite
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from natureqn import NatureQN

from config import atariconfig_teacher as config

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='Environment.')
    parser.add_argument('exp_name', type=str, help='Experiment name.')
    parser.add_argument('state_history', type=int, help='Length of state history (inclusive of current state).')
    parser.add_argument('-pp', '--preprocess', type=str, default=None,
            choices=[None, 'greyscale', 'blackandwhite'], 
            help='Preprocessing function.')
    args = parser.parse_args()

    # set config variables
    config.env_name = args.env_name
    config.output_path = 'results/{0}/teacher'.format(args.exp_name)
    config.student = False
    config.state_history = args.state_history

    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    if args.preprocess is not None:
        env = PreproWrapper(env, prepro=eval(args.preprocess), shape=(80, 80, 1), 
                            overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)