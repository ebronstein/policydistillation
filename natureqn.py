import pdb
import sys

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.dqn_utils import build_mlp
from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def feedforward_nn(self, state, scope, reuse=False):
        num_actions = self.env.action_space.n
        return build_mlp(state, num_actions, scope, self.config.n_layers, 
            self.config.nn_size, activation=self.config.nn_activation, reuse=reuse)

    def nature_cnn(self, state, scope, reuse=False):
        num_actions = self.env.action_space.n
        out = state

        # compress the student network
        size1, size2, size3, size4 = self.q_network_sizes
        # size1, size2, size3, size4 = (16, 16, 16, 128) if self.student else (32, 64, 64, 512)

        # Berkeley Deep RL implementation
        with tf.variable_scope(scope, reuse=reuse):
            # with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=size1, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=size2, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=size3, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.flatten(out)
            # with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=size4,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out

        # original implementation
        # with tf.variable_scope(scope, reuse=reuse):
        #     conv1 = layers.conv3d(inputs=out, num_outputs=size1, kernel_size=[8,8], stride=4) #20
        #     conv2 = layers.conv3d(inputs=conv1, num_outputs=size2, kernel_size=[4,4], stride=2) #10
        #     conv3 = layers.conv3d(inputs=conv2, num_outputs=size3, kernel_size=[3,3], stride=1) #10
        #     hidden = layers.fully_connected(layers.flatten(conv3), size4)
        #     out = layers.fully_connected(hidden, num_actions, activation_fn=None)

        # return out


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions. Overrides Linear get_q_values_op method.

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        if self.config.q_values_model == 'feedforward_nn':
            print('Using feedforward NN for Q-values.')
            return self.feedforward_nn(state, scope, reuse)
        elif self.config.q_values_model == 'nature_cnn':
            print('Using Nature (DeepMind) CNN for Q-values.')
            return self.nature_cnn(state, scope, reuse)
        else:
            print('ERROR: Invalid Q-value model type.')
            sys.exit()

"""
Use deep Q network for test environment.
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
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
