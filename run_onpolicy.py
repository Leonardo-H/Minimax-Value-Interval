import numpy as np
import gym
import pickle
import argparse
import time
import sys
sys.path.append('..')

import utils as U
from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from Logger.logger import ope_log_class

def get_parser():
    parser = argparse.ArgumentParser(description='MWL with PG')
    parser.add_argument('--seed', type = int, default = 1000, help='random seed')
    parser.add_argument('--gamma', type = float, default = 0.99, help='discounted factor')
    parser.add_argument('--tau', type = float, default = 1.0, help='temperature')
    parser.add_argument('--ep-len', type = int, default = 1000, help='horizon length')
    
    args = parser.parse_args()

    return args

def main(args):
    env_name = "CartPole"
    ep_len = args.ep_len
    seed = args.seed
    U.set_seed(seed)

    env = CartPoleEnv(max_ep_len=ep_len, seed=seed + 1000)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    '''load evaluation policy'''
    q_net = Q_network(obs_dim, act_dim, seed=args.seed + 2000, default_tau=args.tau)
    U.initialize_all_vars()
    q_net.load_model('./CartPole_Model/Model')
    
    value_true = U.eval_policy_cartpole(env, q_net, ep_num=150, gamma=args.gamma)

    log_name = 'log.pickle'
    logger = ope_log_class(path='./log/OnPolicy/{}'.format(args.tau), name=log_name, tau=args.tau, env_name=env_name, value_true=value_true)

    logger.dump()

if __name__ == '__main__':
    args = get_parser()
    main(args)
