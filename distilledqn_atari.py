import argparse
import pdb
import time
import sys

import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule, PiecewiseExploration, PiecewiseSchedule
from distilledqn import DistilledQN
from teacher_choice import RandomBandit, EpsilonGreedyBandit

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
    parser.add_argument('choose_teacher_q', 
        choices=['none', 'mean', 'random_bandit', 'eps_greedy_bandit'],
        help='How to choose the teacher Q values for the student loss at each iteration.')
    parser.add_argument('-tcd', '--teacher_checkpoint_dirs', nargs='+', type=str, 
            help='Paths to teachers\' checkpoint files (in same order as their names).')
    parser.add_argument('-tcn', '--teacher_checkpoint_names', nargs='+', type=str, 
            help='Names of the teacher models (in same order as their checkpoint files).')
    parser.add_argument('-tqns', '--teacher_q_network_sizes', type=str, nargs='+',
        choices=['large', 'small'], default='large',
        help='The sizes of the teachers\' Q networks.')
    parser.add_argument('-sqns', '--student_q_network_size', type=str, nargs=1,
        choices=['large', 'small'], default='small',
        help='The size of the student Q network.')
    parser.add_argument('-stqt', '--softmax_teacher_q_tau', type=float, default=0.01,
        help='Value of tau in softmax for processing teacher Q values.')
    parser.add_argument('-wmse', '--mse_prob_loss_weight', type=float, default=1.,
        help='Weight associated with the student loss of MSE over action probabilities.')
    parser.add_argument('-wnll', '--nll_loss_weight', type=float, default=1.,
        help='Weight associated with the student loss of NLL over Q values or action probabilities.')
    parser.add_argument('-nt', '--nsteps_train', type=int, default=5000000,
        help='Number of timesteps to train for.')

    args = parser.parse_args()

    # assertions
    assert len(args.teacher_checkpoint_dirs) == len(args.teacher_checkpoint_names)
    assert len(args.teacher_q_network_sizes) == len(args.teacher_checkpoint_names)

    # get config
    student_config_class = eval('config.{0}_config_student'.format(
            args.env_name.replace('-', '_')))
    # go up 2 directories to 'results/{exp_name}', then save to directory 'student_{exp_name}'
    if len(args.teacher_checkpoint_dirs) == 1:
        output_path = args.teacher_checkpoint_dirs[0] + '../../student_{0}/'.format(args.exp_name)
    else:
        output_path = 'results/student_{0}/'.format(args.exp_name)
    student_config = student_config_class(args.env_name, args.exp_name, 
            output_path, args.nsteps_train, args.teacher_checkpoint_dirs, 
            args.teacher_checkpoint_names)
    
    # set config variables from command-line arguments
    teacher_q_network_sizes = []
    for s in args.teacher_q_network_sizes:
        if s == 'large':
            teacher_q_network_sizes.append((32, 64, 64, 512))
        elif s == 'small':
            teacher_q_network_sizes.append((16, 16, 16, 128))
        else:
            print('"{0}" is not a valid teacher Q network size.'.format(s))
            sys.exit()
    student_config.teacher_q_network_sizes = teacher_q_network_sizes

    if args.student_q_network_size == 'large':
        student_q_network_size = (32, 64, 64, 512)
    elif args.student_q_network_size == 'small':
        student_q_network_size = (16, 16, 16, 128)
    else:
        print('"{0}" is not a valid student Q network size.'.format(args.student_q_network_size))
        sys.exit()
    student_config.student_q_network_size = student_q_network_size
    
    student_config.student_loss = args.student_loss
    student_config.process_teacher_q = args.process_teacher_q
    student_config.choose_teacher_q = args.choose_teacher_q
    student_config.softmax_teacher_q_tau = args.softmax_teacher_q_tau
    student_config.mse_prob_loss_weight = args.mse_prob_loss_weight
    student_config.nll_loss_weight = args.nll_loss_weight
    student_config.nsteps_train = args.nsteps_train
    student_config.lr_nsteps = args.nsteps_train / 2

    # make env
    env = gym.make(student_config.env_name)
    if hasattr(student_config, 'skip_frame'):
        env = MaxAndSkipEnv(env, skip=student_config.skip_frame)
    if hasattr(student_config, 'preprocess_state') and student_config.preprocess_state is not None:
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=student_config.overwrite_render)

    # exploration strategy
    exp_schedule = PiecewiseExploration(env, student_config.exp_endpoints, 
            outside_value=student_config.exp_outside_value)
    # exp_schedule = LinearExploration(env, student_config.eps_begin, 
    #         student_config.eps_end, student_config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = PiecewiseSchedule(student_config.lr_endpoints, 
            outside_value=student_config.lr_outside_value)
    # lr_schedule  = LinearSchedule(student_config.lr_begin, student_config.lr_end,
    #         student_config.lr_nsteps)

    # teacher choice strategy
    if args.choose_teacher_q in ['random_bandit', 'eps_greedy_bandit']:
        eps_schedule = LinearSchedule(student_config.teacher_choice_eps_begin, 
                student_config.teacher_choice_eps_end,
                student_config.teacher_choice_eps_nsteps)
        num_teachers = len(args.teacher_checkpoint_dirs)
    
    if args.choose_teacher_q == 'random_bandit':
        choose_teacher_strategy = RandomBandit(num_teachers, eps_schedule)
    elif args.choose_teacher_q == 'eps_greedy_bandit':
        choose_teacher_strategy = EpsilonGreedyBandit(num_teachers, eps_schedule)
    elif args.choose_teacher_q == 'none':
        choose_teacher_strategy = None
    else:
        raise NotImplementedError

    # train model
    model = DistilledQN(env, student_config)
    model.run(exp_schedule, lr_schedule, choose_teacher_strategy)