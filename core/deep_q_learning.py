import os
import pdb
import numpy as np
import tensorflow as tf
import time

from core.q_learning import QN


class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """
    def add_placeholders_op(self):
        raise NotImplementedError


    def get_q_values_op(self, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError


    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state


    def build(self, student=False):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # define scopes
        q_scope = "q%d" % student
        target_q_scope = "target_q%d" % student

        # compute Q values of state
        s = self.process_state(self.s)
        # self.q = self.get_q_values_op(s, scope="q%d" % student, reuse=False)
        self.q = self.get_q_values_op(s, scope=q_scope, reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        # self.target_q = self.get_q_values_op(sp, scope="target_q%d" % student, reuse=False)
        self.target_q = self.get_q_values_op(sp, scope=target_q_scope, reuse=False)

        # add update operator for target network
        # self.add_update_target_op("q%d" % student, "target_q%d" % student)
        self.add_update_target_op(q_scope, target_q_scope)

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optimizer for the main networks
        # self.add_optimizer_op("q%d" % student)
        # add self.config.exp_name to the scope to properly access the variables
        if self.parent_scope is not None:
            self.add_optimizer_op(self.parent_scope + '/' + q_scope)
        else:
            self.add_optimizer_op(q_scope)


    def initialize_basic(self):
        # create tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 3}
            )
        )

        # TODO: add name argument to restore weights to a unique model name
        # ex:
        # Add ops to save and restore only `v2` using the name "v2"
        # saver = tf.train.Saver({"v2": v2})

        # for saving networks weights
        self.saver = tf.train.Saver()


    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        
        self.initialize_basic()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

       
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)



    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values


    def update_step(self, t, replay_buffer, lr, choose_teacher_strategy=None):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        # if not self.student: # saving states
        #     self.s_batches.append(s_batch)
        #     self.a_batches.append(a_batch)
        #     self.r_batches.append(r_batch)
        #     self.sp_batches.append(sp_batch)
        #     self.done_mask_batches.append(done_mask_batch)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: self.avg_reward, 
            self.max_reward_placeholder: self.max_reward, 
            self.std_reward_placeholder: self.std_reward, 
            self.avg_q_placeholder: self.avg_q, 
            self.max_q_placeholder: self.max_q, 
            self.std_q_placeholder: self.std_q, 
            self.eval_reward_placeholder: self.eval_reward, 
        }

        if self.student:
            # Choose which teacher's Q values to learn from
            if choose_teacher_strategy is not None:
                teacher_idx = choose_teacher_strategy.choose_teacher()
            else:
                # there can only be one teacher if there is no strategy to choose the strategy
                assert self.num_teachers == 1
                teacher_idx = 0
            teachermodel = self.teachermodels[teacher_idx]
            teacher_q = teachermodel.sess.run([teachermodel.q], 
                    feed_dict={teachermodel.s: s_batch})[0]
            fd[self.teacher_q] = teacher_q

            # teacher_q_vals_list = []
            # for teachermodel in self.teachermodels:
            #     teacher_q_vals = teachermodel.sess.run([teachermodel.q], 
            #         feed_dict={teachermodel.s: s_batch})[0]
            #     teacher_q_vals_list.append(teacher_q_vals)
            # fd[self.teacher_q] = teacher_q_vals_list

            # self.teacher_q_idx += 1

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)

