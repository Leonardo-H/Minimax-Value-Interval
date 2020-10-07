import numpy as np
import tensorflow as tf
import time

def make_session(num_cpu=4):
    """Returns a session that will use <num_cpu> CPU's only"""
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        gpu_options=gpu_options)
    return tf.Session(config=tf_config)


def initialize_all_vars():
    tf.get_default_session().run(tf.global_variables_initializer())


def set_seed(seed):
    tf.set_random_seed(seed + 1)
    np.random.seed(seed + 2)


def eval_policy_cartpole(env, alg, ep_num=10, gamma=None, prt=False, save_data=False):
    accum_rew = 0.0
    rew_list = []
    obs_list = []
    act_list = []

    assert hasattr(alg, 'sample_action')
    for i in range(ep_num):
        if prt:
            if i > 0:
                print('Traj ', i)
                print(accum_rew / i)
        obs = env.reset()
        done = False
        factor = 1.0

        while not done:
            act = np.squeeze(alg.sample_action([obs]))

            obs_list.append(obs)
            act_list.append(act)

            obs, rew, done, _ = env.step(act)
            
            rew *= factor
            factor *= gamma

            accum_rew += rew
            rew_list.append(rew)

    if save_data:
        return accum_rew / ep_num, np.array(obs_list), np.array(act_list).reshape([-1, 1])
    else:
        return accum_rew / ep_num


def get_percentile(data):
    ptr = []
    for i in range(10):
        ptr.append(np.percentile(data, i * 10 + 5))
    print(ptr)