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
    parser.add_argument('teacher_checkpoint_dir', type=str, 
            help='Path to teacher checkpoint file.')
    parser.add_argument('student_loss', type=str, choices=['mse', 'nll', 'kl'],
        help='The loss the student uses to learn from the teacher\'s Q values.')
    parser.add_argument('process_teacher_q', type=str, choices=['softmax'],
        help='How to process the teacher Q values for the student loss.')
    parser.add_argument('-stqt', '--softmax_teacher_q_tau', type=float, default=0.01,
        help='Value of tau in softmax for processing teacher Q values.')
    args = parser.parse_args()

    # get config
    student_config_class = eval('config.{0}_config_student'.format(
            args.env_name.replace('-', '_')))
    # go up 2 directories to 'results/{exp_name}', then save to directory 'student_{exp_name}'
    output_path = args.teacher_checkpoint_dir + '../../student_{0}/'.format(args.exp_name)
    student_config = student_config_class(args.env_name, args.exp_name, output_path, args.teacher_checkpoint_dir)
    # set config variables from command-line arguments
    student_config.student_loss = args.student_loss
    student_config.process_teacher_q = args.process_teacher_q
    student_config.softmax_teacher_q_tau = args.softmax_teacher_q_tau

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