import argparse
import pdb
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
    parser.add_argument('student_loss', type=str, 
        choices=['mse_qval', 'mse_prob', 'mse_prob_nll', 'nll', 'kl'],
        help='The loss the student uses to learn from the teacher\'s Q values.')
    parser.add_argument('process_teacher_q', choices=['none', 'softmax_tau'],
        help='How to process the teacher Q values for the student loss.')
    parser.add_argument('choose_teacher_q', choices=['mean'],
        help='How to choose the teacher Q values for the student loss at each iteration.')
    parser.add_argument('-tcd', '--teacher_checkpoint_dirs', nargs='+', type=str, 
            help='Paths to teachers\' checkpoint files (in same order as their names).')
    parser.add_argument('-tcn', '--teacher_checkpoint_names', nargs='+', type=str, 
            help='Names of the teacher models (in same order as their checkpoint files).')
    parser.add_argument('-stqt', '--softmax_teacher_q_tau', type=float, default=0.01,
        help='Value of tau in softmax for processing teacher Q values.')
    parser.add_argument('-wmse', '--mse_prob_loss_weight', type=float, default=1.,
        help='Weight associated with the student loss of MSE over action probabilities.')
    parser.add_argument('-wnll', '--nll_loss_weight', type=float, default=1.,
        help='Weight associated with the student loss of NLL over Q values or action probabilities.')

    args = parser.parse_args()

    # assertions
    assert len(args.teacher_checkpoint_dirs) == len(args.teacher_checkpoint_names)

    # get config
    student_config_class = eval('config.{0}_config_student'.format(
            args.env_name.replace('-', '_')))
    # go up 2 directories to 'results/{exp_name}', then save to directory 'student_{exp_name}'
    if len(args.teacher_checkpoint_dirs) == 1:
        output_path = args.teacher_checkpoint_dirs[0] + '../../student_{0}/'.format(args.exp_name)
    else:
        output_path = 'results/student_{0}/'.format(args.exp_name)
    student_config = student_config_class(args.env_name, args.exp_name, 
            output_path, args.teacher_checkpoint_dirs, 
            args.teacher_checkpoint_names)
    # set config variables from command-line arguments
    student_config.student_loss = args.student_loss
    student_config.process_teacher_q = args.process_teacher_q
    student_config.choose_teacher_q = args.choose_teacher_q
    student_config.softmax_teacher_q_tau = args.softmax_teacher_q_tau
    student_config.mse_prob_loss_weight = args.mse_prob_loss_weight
    student_config.nll_loss_weight = args.nll_loss_weight

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