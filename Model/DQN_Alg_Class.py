import tensorflow as tf
import numpy as np
import time
from functools import partial
from Model.Basic_Alg_Class import Basic_Alg

class DQN():
    def __init__(self, obs_dim, act_dim, *, norm, seed,
                q_hidden_layers=[32, 32], lr=5e-3,
                target_update_freq=1000, ep_len=1000, gamma=0.99, scope='DQN'):
        self.scope = scope

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.norm = norm
        self.seed = seed
        self.q_hidden_layers = q_hidden_layers
        self.lr = lr
        self.tar_update_freq = target_update_freq
        self.ep_len = ep_len
        self.gamma = gamma
        
        self.sess = tf.get_default_session()

        self.trainable_vars = []

        self.build_graph()
        self.build_loss_func()
        self.build_assign()
        self.sess.run(
            [tf.variables_initializer(self.trainable_vars)]
        )
        self.sync()

    def create_value_func(self, obs_tf, act_tf, *, target=False, reuse=False):
        activation = tf.nn.tanh
        # last_activation = lambda x: tf.clip_by_value(x, clip_value_min=-50.0, clip_value_max=300.0)
        last_activation = None
        
        if target:  # build target network      
            with tf.variable_scope(self.scope + 'TargetNet', reuse=reuse):
                x = tf.concat([obs_tf, tf.zeros([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation, trainable=False)
                q0 = tf.layers.dense(x, 1, activation=last_activation, trainable=False,)
            
            with tf.variable_scope(self.scope + 'TargetNet', reuse=True):
                x = tf.concat([obs_tf, tf.ones([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation, trainable=False)
                q1 = tf.layers.dense(x, 1, activation=last_activation, trainable=False,)
            value = tf.cast(q0 > q1, tf.float32) * q0 + tf.cast(q0 <= q1, tf.float32) * q1
            self.target_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope + 'TargetNet')
            
        else:   # not target
            with tf.variable_scope(self.scope + 'QNet', reuse=reuse):
                x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q = tf.layers.dense(x, 1, activation=last_activation, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1.),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(1.))                   
            value = q
            self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + 'QNet') 
            self.trainable_vars += self.q_vars

        return value

    def create_policy(self, obs_tf):
        activation = tf.nn.tanh

        self.obs_pi = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        obs_tf = self.obs_pi

        with tf.variable_scope(self.scope + 'QNet', reuse=True):
            x = tf.concat([obs_tf, tf.zeros([tf.shape(obs_tf)[0], 1])], axis=1)
            for h in self.q_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q0 = tf.layers.dense(x, 1, activation=None,)

        with tf.variable_scope(self.scope + 'QNet', reuse=True):
            x = tf.concat([obs_tf, tf.ones([tf.shape(obs_tf)[0], 1])], axis=1)
            for h in self.q_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q1 = tf.layers.dense(x, 1, activation=None)

        ratio = q1 / q0
        self.logits = tf.concat([q0, q1], axis=1)
        self.prob = tf.nn.softmax(self.logits, axis=1)

        return tf.cast(q1 > q0, tf.int32)
        

    def build_assign(self):
        self.assign_group = []
        for (t_var, q_var) in zip(self.target_vars, self.q_vars):
            self.assign_group.append(tf.assign(t_var, q_var))

    def build_graph(self):
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.done_ph = tf.placeholder(dtype=tf.bool, shape=[None, 1])
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])   
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.normed_factor = self.factor / tf.reduce_mean(self.factor)

        self.tar_func  = (1 - tf.cast(self.done_ph, tf.float32)) \
                            * tf.stop_gradient(self.create_value_func(self.next_obs_ph, None, target=True))
        self.q_func = self.create_value_func(self.obs_ph, self.act_ph)
        self.policy = self.create_policy(self.obs_ph)
        

    def build_loss_func(self):
        self.bellman_error = tf.reduce_mean(self.normed_factor * tf.square(self.rew_ph + self.gamma * self.tar_func - self.q_func))
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.bellman_error)
        self.trainable_vars += self.opt.variables()

    def train(self, data):
        error, _ = self.sess.run(
            [self.bellman_error, self.train_op],
            feed_dict = {        
                self.obs_ph: data['obs'],   
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],            
                self.rew_ph: data['rews'],
                self.done_ph: data['done'],
                self.factor: data['factor'],
            }
        )
        return error

    def sample_action(self, obs):
        if self.norm is not None:
            obs = (obs - self.norm['shift']) / self.norm['scale']
        return self.sess.run(self.policy, feed_dict={self.obs_pi: obs})

    def sync(self):
        self.sess.run(self.assign_group)