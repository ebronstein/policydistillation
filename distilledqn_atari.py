import argparse
import time

import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from distilledqn import DistilledQN

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
    parser.add_argument('teacher_checkpoint_dir', type=str, 
            help='Path to teacher checkpoint file.')
    args = parser.parse_args()

    # get config
    student_config = eval('config.{0}_config_student'.format(
            args.env_name.replace('-', '_')))
    # set config variables
    student_config.env_name = args.env_name
    student_config.output_path = 'results/{0}_{1}/student/'.format(
                args.exp_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    student_config.model_output = student_config.output_path + "model.weights/"
    student_config.log_path     = student_config.output_path + "log.txt"
    student_config.plot_output  = student_config.output_path + "scores.png"
    student_config.record_path  = student_config.output_path + "monitor/"
    student_config.student = True
    student_config.teacher_checkpoint_dir = args.teacher_checkpoint_dir

    # make env
    env = gym.make(student_config.env_name)
    if hasattr(student_config, 'skip_frame'):
        env = MaxAndSkipEnv(env, skip=student_config.skip_frame)
    if hasattr(student_config, 'preprocess_state') and student_config.preprocess_state is not None:
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=student_config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, student_config.eps_begin, 
            student_config.eps_end, student_config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(student_config.lr_begin, student_config.lr_end,
            student_config.lr_nsteps)

    # train model
    model = DistilledQN(env, student_config)
    model.run(exp_schedule, lr_schedule)