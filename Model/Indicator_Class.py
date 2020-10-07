import tensorflow as tf
import numpy as np
import time
from functools import partial

class Indicator(object):
    def __init__(self, obs_dim, n_func=50, 
                    mean=0.0, sigma=1.0,
                    hidden_layers=[32, 32]):
        self.obs_dim = obs_dim
        self.n_func = n_func
        self.mean = mean
        self.sigma = sigma
        self.hidden_layers = hidden_layers

        self.sess = tf.get_default_session()

        self.scope = 'indicator'

        self.build_graph()
        self.sess.run([tf.variables_initializer(tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope))])


    def generate_random_function(self, obs_tf, act_tf, factor_tf):
        activation = tf.tanh
        last_activation = None
        self.func_list = []
        obs_act = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
        with tf.variable_scope(self.scope, reuse=False):
            for n in range(self.n_func):
                x = obs_act
                for h in self.hidden_layers:
                    x = tf.layers.dense(x, h, activation=activation, trainable=False)
                q = tf.layers.dense(x, 1, activation=last_activation, trainable=False) * factor_tf
                self.func_list.append(tf.reduce_mean(tf.stop_gradient(q)))

        
    def build_graph(self):
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.factor = tf.placeholder(dtype=tf.float32, shape=[None, 1])       
        self.generate_random_function(self.obs_ph, self.act_ph, self.factor)


    def infer(self, obs, act, factor):
        func_list = self.sess.run(
            self.func_list,
            feed_dict={
                self.obs_ph: obs,
                self.act_ph: act,
                self.factor: factor,
            }
        )
        return np.array(func_list).reshape([-1])