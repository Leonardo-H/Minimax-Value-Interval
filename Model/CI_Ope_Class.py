import tensorflow as tf
import numpy as np
import time
from functools import partial
from Model.Basic_Alg_Class import Basic_Alg

class CI_SGDA(Basic_Alg):
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
        self.debug = {}

        self.q_net = q_net
        self.norm = norm

        self.scope_dict = {
            'inf_w': self.scope + '_inf_w',
            'sup_w': self.scope + '_sup_w',
            'inf_q': self.scope + '_inf_q',
            'sup_q': self.scope + '_sup_q',
        }

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
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])          
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])

        # inf q & sup w
        ''' extended estimation for v(s_0) '''
        self.inf_q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False, scope=self.scope_dict['inf_q'])        
        self.inf_v_init = tf.reduce_mean(self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['inf_q']))
        self.inf_v_next = self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['inf_q'])        
        
        self.sup_w = self.create_w_func(self.obs_ph, self.act_ph, factor=self.factor, reuse=False, scope=self.scope_dict['sup_w'])

        # sup q & inf w
        self.sup_q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False, scope=self.scope_dict['sup_q'])        
        self.sup_v_init = tf.reduce_mean(self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['sup_q']))
        self.sup_v_next = self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True, scope=self.scope_dict['sup_q'])
        
        self.inf_w = self.create_w_func(self.obs_ph, self.act_ph, factor=self.factor, reuse=False, scope=self.scope_dict['inf_w'])
        

    def create_w_func(self, obs_tf, act_tf, *, factor, reuse=False, scope=None):
        assert scope is not None
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
            for h in self.w_hidden_layers:
                x = tf.layers.dense(x, h, activation=tf.nn.tanh)
            w = tf.layers.dense(x, 1, activation=None)

            w = tf.abs(w) * factor

            if reuse == False:
                self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                
            return w

    def create_value_func(self, obs_tf, act_tf, *, func_type, reuse=False, scope=None):
        assert scope is not None
        activation = tf.nn.tanh
        last_activation = lambda x: tf.clip_by_value(x, clip_value_min=-50.0, clip_value_max=300.0)

        if func_type == 'v':
            if self.norm['type'] is not None:
                org_obs = obs_tf * self.norm['scale'] + self.norm['shift']
            else:
                org_obs = obs_tf
            prob_mask = self.q_net.build_prob(org_obs, split=True)
            
            with tf.variable_scope(scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.zeros([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q0 = tf.layers.dense(x, 1, activation=last_activation)
            
            with tf.variable_scope(scope, reuse=True):
                x = tf.concat([obs_tf, tf.ones([tf.shape(obs_tf)[0], 1])], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q1 = tf.layers.dense(x, 1, activation=last_activation)
            value = q0 * prob_mask[0] + q1 * prob_mask[1]
        else:
            with tf.variable_scope(scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
                for h in self.q_hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation)
                q = tf.layers.dense(x, 1, activation=last_activation)                   
                value = q

        if reuse == False:
            self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) 

        return value

    def build_loss_func(self):
        epsilon = 1e-9 # avoid divide 0

        ''' 
            inf_q sup_w q(s_0, a) + \EE[...]
        '''
        self.sup_w_error = self.rew_ph + self.gamma * self.inf_v_next - self.inf_q
        self.sup_w_mask = tf.cast(self.sup_w_error > 0, tf.float32)
        
        mean = epsilon + tf.reduce_mean(self.sup_w * self.sup_w_mask + tf.stop_gradient(self.sup_w) * (1 - self.sup_w_mask))
        self.sup_w = self.sup_w * self.sup_w_mask / mean * self.scale
        self.debug.update({"sup w mask > 0": tf.reduce_sum(tf.cast(self.sup_w_mask > 0.0, tf.float32))})
        self.debug.update({"mean sup w mask": tf.reduce_mean(self.sup_w_mask)})
    
        self.mean_error_upper = tf.reduce_mean(self.sup_w_error * self.sup_w)

        self.inf_q_loss = tf.squeeze(self.inf_v_init / self.ep_len + self.mean_error_upper)
        self.inf_q_opt = tf.train.AdamOptimizer(self.q_lr)
        self.inf_q_train_op = self.inf_q_opt.minimize(self.inf_q_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['inf_q']))
        self.trainable_vars += self.inf_q_opt.variables()

        self.sup_w_loss = -self.inf_q_loss
        self.sup_w_opt = tf.train.AdamOptimizer(self.w_lr)        
        self.sup_w_train_op = self.sup_w_opt.minimize(self.sup_w_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['sup_w']))
        self.trainable_vars += self.sup_w_opt.variables()

        ''' 
            sup_q inf_w q(s_0, a) + \EE[...]
        '''
        self.inf_w_error = self.rew_ph + self.gamma * self.sup_v_next - self.sup_q      
        self.inf_w_mask = tf.cast(self.inf_w_error < 0, tf.float32)
        
        mean = epsilon + tf.reduce_mean(self.inf_w * self.inf_w_mask + tf.stop_gradient(self.inf_w) * (1 - self.inf_w_mask))
        self.inf_w = self.inf_w * self.inf_w_mask / mean * self.scale
        self.debug.update({"inf w mask > 0": tf.reduce_sum(tf.cast(self.inf_w_mask > 0.0, tf.float32))})
                
        self.mean_error_lower = tf.reduce_mean(self.inf_w_error * self.inf_w)

        self.sup_q_loss = -tf.squeeze(self.sup_v_init / self.ep_len + self.mean_error_lower)
        self.sup_q_opt = tf.train.AdamOptimizer(self.q_lr)
        self.sup_q_train_op = self.sup_q_opt.minimize(self.sup_q_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['sup_q']))
        self.trainable_vars += self.sup_q_opt.variables()

        self.inf_w_loss = -self.sup_q_loss
        self.inf_w_opt = tf.train.AdamOptimizer(self.w_lr)        
        self.inf_w_train_op = self.inf_w_opt.minimize(self.inf_w_loss, var_list=self.get_all_vars_with_scope(self.scope_dict['inf_w']))
        self.trainable_vars += self.inf_w_opt.variables()        


    def build_estimation_graph(self):
        self.upper_bound = self.inf_q_loss * self.ep_len
        self.lower_bound = -self.sup_q_loss * self.ep_len        
        self.value_estimation = (self.upper_bound + self.lower_bound) / 2.0

    def train_w(self, data_sup, data_inf):
        data = data_sup
        sup_w_loss, _ = tf.get_default_session().run(
            [self.sup_w_loss, self.sup_w_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        data = data_inf
        inf_w_loss, _ = tf.get_default_session().run(
            [self.inf_w_loss, self.inf_w_train_op],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return sup_w_loss, inf_w_loss

    def train_q(self, data_sup, data_inf):
        data = data_sup
        sup_q_loss, _, sup_debug = tf.get_default_session().run(
            [self.sup_q_loss, self.sup_q_train_op, self.debug],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        data = data_inf
        inf_q_loss, _, inf_debug = tf.get_default_session().run(
            [self.inf_q_loss, self.inf_q_train_op, self.debug],
            feed_dict={
                self.obs_ph: data['obs'],
                self.next_obs_ph: data['next_obs'],
                self.act_ph: data['acts'],              
                self.rew_ph: data['rews'],
                self.init_obs_ph: data['init_obs'],
                self.factor: data['factor'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return sup_q_loss, inf_q_loss, sup_debug, inf_debug

    def evaluation(self, dataset):        
        lower_q, upper_q, lower_bound, upper_bound = tf.get_default_session().run(
            [self.sup_v_init, self.inf_v_init, self.lower_bound, self.upper_bound],
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
        return lower_q, upper_q, lower_bound, upper_bound
