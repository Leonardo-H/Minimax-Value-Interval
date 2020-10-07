import tensorflow as tf
import numpy as np
import time
from functools import partial
from Model.Basic_Alg_Class import Basic_Alg

class MQL_SGDA(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, norm, q_net,
                       q_hidden_layers=[32, 32], q_lr=5e-3, 
                       w_hidden_layers=[32, 32], w_lr=5e-3, 
                       scope='sgda', scale=0.1,
                       ep_len=1000, gamma=0.99):
        super().__init__(scope)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.scale = scale
        self.ep_len = ep_len
        self.q_lr = q_lr
        self.q_hidden_layers = q_hidden_layers
        self.w_lr = w_lr
        self.w_hidden_layers = w_hidden_layers

        self.q_net = q_net
        self.norm = norm

        self.trainable_vars = []

        self.build_graph()
        self.build_loss_func()
        self.build_estimation_graph()

        tf.get_default_session().run(
            [tf.variables_initializer(self.trainable_vars)]
        )
    
    def build_graph(self):     
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_act = tf.concat([self.obs_ph, tf.cast(self.act_ph, tf.float32)], axis=1)
        self.q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False)
        self.w = self.create_w_func(self.obs_ph, self.act_ph, factor=self.factor, reuse=False)

        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])  
        self.v_next = self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True)


    def create_w_func(self, obs_tf, act_tf, *, factor, reuse=False):
        with tf.variable_scope(self.scope + '_w_func', reuse=reuse):
            x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
            for h in self.w_hidden_layers:
                x = tf.layers.dense(x, h, activation=tf.nn.tanh)
            w = tf.layers.dense(x, 1, activation=None)
            w = tf.abs(w) * factor

            if reuse == False:
                self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '_w_func')
                
            return w

    def create_value_func(self, obs_tf, act_tf, *, func_type, reuse=False, normalize=True):
        activation = tf.nn.tanh
        last_activation = lambda x: tf.clip_by_value(x, clip_value_min=-20.0, clip_value_max=300.0)

        if func_type == 'v':
            if self.norm['type'] is not None:
                org_obs = obs_tf * self.norm['scale'] + self.norm['shift']
            else:
                org_obs = obs_tf
            prob_mask = self.q_net.build_prob(org_obs, split=True)
            
            with tf.variable_scope(self.scope + '_q_func', reuse=reuse):
                x = tf.concat([obs_tf, tf.zeros([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q0 = tf.layers.dense(x, 1, activation=last_activation)
            
            with tf.variable_scope(self.scope + '_q_func', reuse=True):
                x = tf.concat([obs_tf, tf.ones([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q1 = tf.layers.dense(x, 1, activation=last_activation)
            value = q0 * prob_mask[0] + q1 * prob_mask[1]
        else:            
            with tf.variable_scope(self.scope + '_q_func', reuse=reuse):
                x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q = tf.layers.dense(x, 1, activation=last_activation)
                value = q

        if reuse == False:
            self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '_q_func')                    
  
        return value

    def build_estimation_graph(self):
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.value_estimation = tf.reduce_mean(self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True))
        
        self.lower_bound = self.value_estimation - self.q_loss * self.ep_len
        self.upper_bound = self.value_estimation + self.q_loss * self.ep_len

    def build_loss_func(self):
        epsilon = 1e-9
        self.error = self.rew_ph + self.gamma * self.v_next - self.q
        
        pos_mask = tf.cast(self.error > 0, tf.float32)      # mask negative components
        neg_mask = tf.cast(self.error < 0, tf.float32)      # mask positive components

        mean = epsilon + tf.reduce_mean(self.w * pos_mask + tf.stop_gradient(self.w) * (1 - pos_mask))
        w_pos = self.w * pos_mask / mean * self.scale
        self.loss_pos = tf.abs(tf.reduce_mean(self.error * w_pos))

        mean = epsilon + tf.reduce_mean(self.w * neg_mask + tf.stop_gradient(self.w) * (1 - neg_mask))
        w_neg = self.w * neg_mask / mean * self.scale 
        self.loss_neg = tf.abs(tf.reduce_mean(self.error * w_neg))

        def pos_f(): 
            return self.loss_pos

        def neg_f(): 
            return self.loss_neg
                
        self.q_loss = tf.cond(tf.greater(self.loss_pos, self.loss_neg), true_fn=pos_f, false_fn=neg_f)
        self.w_loss = -self.q_loss

        self.q_opt = tf.train.AdamOptimizer(self.q_lr)
        self.w_opt = tf.train.AdamOptimizer(self.w_lr)
        self.q_train_op = self.q_opt.minimize(self.q_loss, var_list=self.get_all_vars_with_scope(self.scope + '_q_func'))
        self.w_train_op = self.w_opt.minimize(self.w_loss, var_list=self.get_all_vars_with_scope(self.scope + '_w_func'))

        self.trainable_vars += self.q_opt.variables()
        self.trainable_vars += self.w_opt.variables()


    def train_q(self, data):
        loss, _ = tf.get_default_session().run(
            [self.q_loss, self.q_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )

        return loss

    def train_w(self, data):
        loss, _ = tf.get_default_session().run(
            [self.w_loss, self.w_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )

        return loss

    def evaluation(self, dataset):        
        value, lower_bound, upper_bound = tf.get_default_session().run(
            [self.value_estimation, self.lower_bound, self.upper_bound],
            feed_dict={
                self.obs_ph: dataset['obs'],   
                self.next_obs_ph: dataset['next_obs'],
                self.act_ph: dataset['acts'],            
                self.rew_ph: dataset['rews'],
                self.init_obs_ph: dataset['init_obs'],
                self.factor: dataset['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value, lower_bound, upper_bound