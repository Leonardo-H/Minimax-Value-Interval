import numpy as np
import gym
import pickle
import argparse
import time
import sys
import multiprocessing

import utils as U
from Model.MQL_SGDA_Class import MQL_SGDA
from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from Logger.logger import ope_log_class
from multiprocessing import Pool
from copy import deepcopy


def get_parser():
    parser = argparse.ArgumentParser(description='MQL_SGDA')
    parser.add_argument('--iter', type = int, default = 200, help='training iteration')
    parser.add_argument('--q-iter', type = int, default = 500, help='training iteration for Q')
    parser.add_argument('--w-iter', type = int, default = 50, help='training iteration for W')

    parser.add_argument('--q-lr', type = float, default = 5e-3, help='learning rate')
    parser.add_argument('--w-lr', type = float, default = 5e-3, help='learning rate')
    parser.add_argument('--bs', type = int, default = 500, help='batch size')  
    parser.add_argument('--gamma', type = float, default = 0.99, help='discounted factor')
    
    parser.add_argument('--norm', type = str, default = 'std_norm', help='normalization type')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed to generate dataset')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
    parser.add_argument('--tau', type = float, nargs='+', default = [1.5], help='policy temperature')
    parser.add_argument('--n-ros', type = int, nargs='+', default = [200], help='# trajectories in dataset')   

    parser.add_argument('--scale', type = float, default = 2.5, help='expecated discounted w(s,a)')
    
    parser.add_argument('--bootstrap', action='store_true', help='whether to use bootstrapping')
    args = parser.parse_args()

    return args

def sample_data(dataset, sample_num):
    data_size = dataset['obs'].shape[0]
    index = np.random.choice(data_size, sample_num)

    return {
        'obs': dataset['obs'][index],
        'next_obs': dataset['next_obs'][index],
        'acts': dataset['acts'][index],
        'rews': dataset['rews'][index],
        'factor': dataset['factor'][index],
    }

def shuffle_data(dataset):
    data_size = dataset['obs'].shape[0]
    init_size = dataset['init_obs'].shape[0]
    index = np.random.choice(data_size, data_size)
    init_index = np.random.choice(init_size, init_size)

    return {
        'obs': dataset['obs'][index],
        'next_obs': dataset['next_obs'][index],
        'acts': dataset['acts'][index],
        'rews': dataset['rews'][index],
        'init_obs': dataset['init_obs'][init_index],
        'factor': dataset['factor'][index],
    }

def main(args):
    command = sys.executable + " " + " ".join(sys.argv)
    env_name = "CartPole"
    ep_len = args.ep_len
    dataset_seed = args.dataset_seed
    seed = args.seed

    U.set_seed(dataset_seed + seed)

    env = CartPoleEnv(max_ep_len=ep_len, seed=seed)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    '''load evaluation policy'''
    q_net = Q_network(obs_dim, act_dim, seed=100, default_tau=args.tau)
    U.initialize_all_vars()
    q_net.load_model('./CartPole_Model/Model')

    file_name = './Dataset/{}/CartPole-ep1000-tau1.0-n{}-seed{}.pickle'.format(args.n_ros, args.n_ros, args.dataset_seed)

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
        norm_type = args.norm
        gamma = args.gamma
        if 'factor' not in dataset.keys():
            dataset['factor'] = np.array([gamma ** (i % 1000) for i in range(dataset['obs'].shape[0])]).reshape([-1, 1])
        if args.bootstrap:
            dataset = shuffle_data(dataset)

        if norm_type == 'std_norm':
            obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            obs_std = np.std(dataset['obs'], axis=0, keepdims=True)

            dataset['obs'] = (dataset['obs'] - obs_mean) / obs_std
            dataset['next_obs'] = (dataset['next_obs'] - obs_mean) / obs_std
            dataset['init_obs'] = (dataset['init_obs'] - obs_mean) / obs_std
            
            norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}
        else:
            norm = {'type': None, 'shift': None, 'scale': None}
            
        print('norm ', norm)


    sgda = MQL_SGDA(obs_dim, act_dim, q_net=q_net,
            q_hidden_layers=[32, 32], q_lr=args.q_lr, w_hidden_layers=[32], w_lr=args.w_lr, 
            scale=args.scale, gamma=args.gamma, ep_len=ep_len, norm=norm)
    sess.graph.finalize()

    value_true = U.eval_policy_cartpole(env, q_net, ep_num=10, gamma=args.gamma, prt=True)

    log_name = 'log_seed{}.pickle'.format(args.seed)

    if args.bootstrap:
        mid_dir = 'bootstrap_log'
    else:
        mid_dir = 'log'
    logger = ope_log_class(path='./{}/{}/MQL_Interval/{}/Dataset{}/{}'.format(mid_dir, args.n_ros, args.scale, args.dataset_seed, args.tau), name=log_name, tau=args.tau, env_name=env_name, value_true=value_true)

    w_prt_interval = 50
    q_prt_interval = 100
    base = 0
    for iter in range(args.iter):
        for i in range(1, args.w_iter + 1):
            data = sample_data(dataset, args.bs)
            w_loss = sgda.train_w(data)

            if i % w_prt_interval == 0 and i != 0:
                print('-------------------------------------')
                print('Iter ', base + i)
                print('Loss ', -w_loss)      
                value_est, lower_bound, upper_bound = sgda.evaluation(dataset)
                print('True value: {}. \nEst Lower Bound: {}\nEst Upper Bound:{}'.format(value_true, lower_bound, upper_bound))

                print('-------------------------------------\n\n')
                logger.update_bound_info(base + i, lower_bound, upper_bound)

        base += args.w_iter

        for i in range(1, args.q_iter + 1):
            data = sample_data(dataset, args.bs)
            q_loss = sgda.train_q(data)

            if i % q_prt_interval == 0 and i != 0:
                print('-------------------------------------')
                print('Iter ', base + i)
                print('Loss ', q_loss)
                value_est, lower_bound, upper_bound = sgda.evaluation(dataset)

                print('True value: {}. \nEst Lower Bound: {}\nEst Upper Bound:{}'.format(value_true, lower_bound, upper_bound))                
                
                print('-------------------------------------\n\n')
                logger.update_bound_info(base + i, lower_bound, upper_bound)
        base += args.q_iter

        if iter % 10 == 0:
            logger.dump()


if __name__ == '__main__':
    args = get_parser()

    args_list = []
    for dataset_seed in args.dataset_seed:
        for tau in args.tau:
            for seed in args.seed:
                for n_ros in args.n_ros:
                    args_copy = deepcopy(args)
                    args_copy.dataset_seed = dataset_seed
                    args_copy.seed = seed
                    args_copy.tau = tau
                    args_copy.n_ros = n_ros
                    args_list.append(args_copy)

    with Pool(processes=len(args_list), maxtasksperchild=1) as p:
            p.map(main, args_list, chunksize=1)

