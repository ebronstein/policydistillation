import pdb

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear
from natureqn import NatureQN

import logging

logging.basicConfig(level=logging.INFO)

# CKPT_DIR = 'results/atari/teacher/model.weights/'

def initialize_teacher(session, model, train_dir, seed=42):
    tf.set_random_seed(seed)
    logging.info("Reading model parameters from %s" % train_dir)
    model.saver.restore(session, train_dir)
    logging.info('[Teacher] Num params: %d' % model.size)

class DistilledQN(NatureQN):
    def __init__(self, env, config, logger=None, student=True):
        teachermodel = NatureQN(env, config)
        teachermodel.initialize_basic()
        initialize_teacher(teachermodel.sess, teachermodel, 
                           config.teacher_checkpoint_dir)
        self.teachermodel = teachermodel
        super(DistilledQN, self).__init__(
            env, config, logger=logger, student=student)

    def add_loss_op(self, q, target_q):
        eps = 0.00001

        if self.config.process_teacher_q == 'softmax_tau':
            # Divide the teacher Q values by tau, which will result in a softmax
            # of a different temperature when a softmax is applied.
            teacher_q = self.teacher_q / self.config.softmax_teacher_q_tau
        elif self.config.process_teacher_q == 'none': # no processing
            teacher_q = self.teacher_q
        else:
            print('"{0}" is not a valid way to proess the teacher Q values'.format(
                    self.config.process_teacher_q))
            sys.exit()

        ##############################

        # Loss functions for distillation

        # Action probabilities (only necessary for certain loss functions)
        # Get the action probabilities of the teacher by applying a softmax
        # with a (potentially different) temperature parameter.
        teacher_prob = tf.nn.softmax(teacher_q, axis=1) + eps
        # Get the action probabilities of the student by applying a softmax.
        prob = tf.nn.softmax(q, axis=1) + eps

        if self.config.student_loss == 'mse_qval':
            # MSE of teacher and student Q values
            self.loss = tf.losses.mean_squared_error(q, teacher_q)
        elif self.config.student_loss == 'mse_prob':
            # MSE of teacher and student action probabilities
            self.loss = tf.losses.mean_squared_error(prob, teacher_prob)
        elif self.config.student_loss == 'nll':
            # NLL of action probabilities
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.argmax(self.teacher_q, axis=1), 
                logits=q))
        elif self.config.student_loss == 'mse_prob_nll':
            # MSE of teacher and student action probabilities
            mse_prob_loss = tf.losses.mean_squared_error(prob, teacher_prob)
            # NLL of action probabilities
            nll_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.argmax(self.teacher_q, axis=1), 
                logits=q))
            self.loss = (self.config.mse_prob_loss_weight * mse_prob_loss + 
                         self.config.nll_loss_weight * nll_loss)
        elif self.config.student_loss == 'kl':
            # KL of action probabilities
            self.loss = tf.reduce_sum(teacher_prob * tf.log(teacher_prob / prob))
        else:
            print('"{0}" is not a valid student loss'.format(self.config.student_loss))
            sys.exit()



"""
Use distilled Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    studentmodel = DistilledQN(env, config)
    studentmodel.run(exp_schedule, lr_schedule)
