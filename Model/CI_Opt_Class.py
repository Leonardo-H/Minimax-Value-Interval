import tensorflow as tf
import numpy as np
import time
import os
from functools import partial
from Model.Basic_Alg_Class import Basic_Alg

class CI_Opt(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, norm, seed,
                       q_hidden_layers=[32, 32], q_lr=5e-3, 
                       pi_hidden_layers=[32, 32], pi_lr=5e-3,
                       scope='sgda', scale=2.5,
                       default_tau=1.0, pi_type='lower',
                       ep_len=1000, gamma=0.99):
        super().__init__(scope)


        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.scale = scale
        self.ep_len = ep_len
        self.q_lr = q_lr
        self.q_hidden_layers = q_hidden_layers
        self.pi_lr = pi_lr
        self.pi_hidden_layers = pi_hidden_layers
        self.default_tau = default_tau        
        self.seed = seed
        self.pi_type = pi_type

        self.norm = norm
        self.sess = tf.get_default_session()

        self.scope_dict = {
            'inf_q': self.scope + '_inf_q',
            'sup_q': self.scope + '_sup_q',
        }

        self.trainable_vars = []

        self.build_pi_network()
        self.build_graph()
        self.build_loss_func()
        self.build_estimation_graph()

        self.sess.run(
            [tf.variables_initializer(self.trainable_vars)]
        )

    def build_pi_network(self):
        self.tau_ph = self.default_tau
        self.obs_pi = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])

        activation = tf.nn.tanh
        last_activation = None
        with tf.variable_scope('policy', reuse=False):
            x = tf.concat([self.obs_pi, tf.zeros([tf.shape(self.obs_pi)[0], 1])], axis=1)
            for h in self.pi_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q0 = tf.layers.dense(x, 1, activation=last_activation)

        with tf.variable_scope('policy', reuse=True):
            x = tf.concat([self.obs_pi, tf.ones([tf.shape(self.obs_pi)[0], 1])], axis=1)
            for h in self.q_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q1 = tf.layers.dense(x, 1, activation=last_activation)

        self.logits = tf.concat([q0, q1], axis=1) / self.tau_ph
        self.prob = tf.nn.softmax(self.logits, axis=1)
        self.random_action = tf.multinomial(self.logits, 1, seed=self.seed)
        self.trainable_vars += self.get_all_vars_with_scope('policy')

    def build_pi_prob(self, obs_ph, reuse=True, split=True):
        activation = tf.nn.tanh
        last_activation = None
        with tf.variable_scope('policy', reuse=True):
            x = tf.concat([obs_ph, tf.zeros([tf.shape(obs_ph)[0], 1])], axis=1)
            for h in self.pi_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q0 = tf.layers.dense(x, 1, activation=last_activation)

        with tf.variable_scope('policy', reuse=True):
            x = tf.concat([obs_ph, tf.ones([tf.shape(obs_ph)[0], 1])], axis=1)
            for h in self.q_hidden_layers:
                x = tf.layers.dense(x, h, activation=activation)
            q1 = tf.layers.dense(x, 1, activation=last_activation)

        logits = tf.concat([q0, q1], axis=1) / self.tau_ph
        prob = tf.nn.softmax(logits, axis=1)

        if split:
            return tf.split(prob, 2, axis=1)
        else:
            return prob

    def sample_action(self, obs):
        if self.norm is not None:
            obs = (obs - self.norm['shift']) / self.norm['scale']
        return self.sess.run(self.random_action, feed_dict={self.obs_pi: obs})
    
    def build_graph(self): 
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])          
        self.done = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.normed_factor = self.factor / tf.reduce_mean(self.factor)
        self.obs_act = tf.concat([self.obs_ph, tf.cast(self.act_ph, tf.float32)], axis=1)
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])          
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])

        # inf q & sup w
        self.inf_q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False, scope=self.scope_dict['inf_q'])        
        self.inf_v_init = tf.reduce_mean(self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['inf_q']))
        self.inf_v_next = (1 - self.done) * self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['inf_q'])        
                
        # sup q & inf w
        self.sup_q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False, scope=self.scope_dict['sup_q'])        
        self.sup_v_init = tf.reduce_mean(self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['sup_q']))
        self.sup_v_next = (1 - self.done) * self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['sup_q'])
        

    def create_value_func(self, obs_tf, act_tf, *, func_type, reuse=False, scope=None):
        assert scope is not None
        activation = tf.nn.tanh
        last_activation = lambda x: tf.clip_by_value(x, clip_value_min=-50.0, clip_value_max=300.0)

        if func_type == 'v':
            prob_mask = self.build_pi_prob(obs_tf, reuse=True, split=True)
            
            with tf.variable_scope(scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.zeros([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q0 = tf.layers.dense(x, 1, activation=last_activation, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1.),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(1.))
            
            with tf.variable_scope(scope, reuse=True):
                x = tf.concat([obs_tf, tf.ones([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q1 = tf.layers.dense(x, 1, activation=last_activation, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1.),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(1.))
            value = q0 * prob_mask[0] + q1 * prob_mask[1]
        else:
            with tf.variable_scope(scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q = tf.layers.dense(x, 1, activation=last_activation, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1.),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(1.))                   
                value = q

        if reuse == False:
            self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) 

        return value

    def build_loss_func(self):
        epsilon = 1e-4 # avoid divide 0

        ''' 
        Lower Bound:
            sup_w inf_q q(s_0, a) + \EE[...]
        '''
        
        self.inf_error = tf.reduce_mean(
            tf.abs(self.normed_factor * (self.rew_ph + self.gamma * self.inf_v_next - self.inf_q))
        )
        self.inf_q_loss = tf.squeeze(self.inf_v_init / self.ep_len + self.scale * self.inf_error)
        self.inf_q_opt = tf.train.AdamOptimizer(self.q_lr)
        self.inf_q_train_op = self.inf_q_opt.minimize(self.inf_q_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['inf_q']))
        self.trainable_vars += self.inf_q_opt.variables()

        ''' 
        Upper Bound:
            inf_w sup_q q(s_0, a) + \EE[...]
        '''        
        self.sup_error = tf.reduce_mean(
            tf.abs(self.normed_factor * (self.rew_ph + self.gamma * self.sup_v_next - self.sup_q))
        )
        self.sup_q_loss = tf.squeeze(self.sup_v_init / self.ep_len - self.scale * self.sup_error)
        self.sup_q_opt = tf.train.AdamOptimizer(self.q_lr)
        self.sup_q_train_op = self.sup_q_opt.minimize(-self.sup_q_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['sup_q']))
        self.trainable_vars += self.sup_q_opt.variables()

        self.pi_opt = tf.train.AdamOptimizer(self.pi_lr)
        if self.pi_type == 'lower':
            self.pi_train_op = self.pi_opt.minimize(-self.inf_q_loss, var_list=self.get_all_vars_with_scope('policy'))
        else:
            self.pi_train_op = self.pi_opt.minimize(-self.sup_q_loss, var_list=self.get_all_vars_with_scope('policy'))
        self.trainable_vars += self.pi_opt.variables()


    def build_estimation_graph(self):
        self.upper_bound = self.sup_q_loss * self.ep_len
        self.lower_bound = self.inf_q_loss * self.ep_len        
        self.value_estimation = (self.upper_bound + self.lower_bound) / 2.0

    def train_q(self, data_sup, data_inf):
        data = data_sup
        sup_q_loss, _ = self.sess.run(
            [self.sup_q_loss, self.sup_q_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.done: data['done'],
            }
        )
        data = data_inf
        inf_q_loss, _ = self.sess.run(
            [self.inf_q_loss, self.inf_q_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.done: data['done'],
            }
        )
        return sup_q_loss, inf_q_loss

    def train_pi(self, dataset):
        upper_bound, lower_bound, _ = self.sess.run(
                [self.upper_bound, self.lower_bound, self.pi_train_op],
                feed_dict={                 
                    self.obs_ph: dataset['obs'],   
                    self.next_obs_ph: dataset['next_obs'],
                    self.act_ph: dataset['acts'],            
                    self.rew_ph: dataset['rews'],
                    self.init_obs_ph: dataset['init_obs'],
                    self.factor: dataset['factor'],
                    self.done: dataset['done'],
                }
            )
        return upper_bound, lower_bound

    def evaluation(self, dataset):        
        lower_q, upper_q, lower_bound, upper_bound = self.sess.run(
            [self.inf_v_init, self.sup_v_init, self.lower_bound, self.upper_bound],
            feed_dict={
                self.obs_ph: dataset['obs'],   
                self.next_obs_ph: dataset['next_obs'],
                self.act_ph: dataset['acts'],            
                self.rew_ph: dataset['rews'],
                self.init_obs_ph: dataset['init_obs'],
                self.factor: dataset['factor'],
                self.done: dataset['done'],
            }
        )
        return lower_q, upper_q, lower_bound, upper_bound

    def get_pi(self, obs):
        return self.sess.run(self.prob, feed_dict={self.obs_pi: obs})