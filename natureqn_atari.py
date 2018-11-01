import argparse
import pdb
import time

import gym
from utils.preprocess import greyscale, blackandwhite
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from natureqn import NatureQN

import config

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
    # parser.add_argument('state_history', type=int, help='Length of state history (inclusive of current state).')
    parser.add_argument('-pp', '--preprocess', type=str, default=None,
            choices=[None, 'greyscale', 'blackandwhite'], 
            help='Preprocessing function.')
    args = parser.parse_args()

    # get config
    teacher_config = eval('config.{0}_config_teacher'.format(
            args.env_name.replace('-', '_')))
    # set config variables
    teacher_config.env_name = args.env_name
    teacher_config.output_path = 'results/{0}_{1}/teacher'.format(
                args.exp_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    teacher_config.model_output = teacher_config.output_path + "model.weights/"
    teacher_config.log_path     = teacher_config.output_path + "log.txt"
    teacher_config.plot_output  = teacher_config.output_path + "scores.png"
    teacher_config.record_path  = teacher_config.output_path + "monitor/"
    teacher_config.student = False
    # teacher_config.state_history = args.state_history

    pdb.set_trace()

    # make env
    env = gym.make(teacher_config.env_name)
    if hasattr(teacher_config, 'skip_frame'):
        env = MaxAndSkipEnv(env, skip=teacher_config.skip_frame)
    if args.preprocess is not None:
        env = PreproWrapper(env, prepro=eval(args.preprocess), shape=(80, 80, 1), 
                            overwrite_render=teacher_config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, teacher_config.eps_begin, 
            teacher_config.eps_end, teacher_config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(teacher_config.lr_begin, teacher_config.lr_end,
            teacher_config.lr_nsteps)

    # train model
    model = NatureQN(env, teacher_config)
    model.run(exp_schedule, lr_schedule)