import numpy as np
import gym
import pickle
import argparse
import time
import sys
import multiprocessing

import utils as U
from Model.DQN_Alg_Class import DQN
from Model.Indicator_Class import Indicator
from Env.CartPole import CartPoleEnv
from Logger.logger import dqn_log_class
from multiprocessing import Pool
from copy import deepcopy


def get_parser():
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--iter', type = int, default = 500000, help='training iteration for DQN')
    parser.add_argument('--lr', type = float, default = 5e-3, help='learning rate')
    parser.add_argument('--bs', type = int, default = 500, help='batch size')  
    parser.add_argument('--gamma', type = float, default = 0.99, help='discounted factor')
    parser.add_argument('--norm', type = str, default = 'std_norm', help='normalization type')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')
    parser.add_argument('--tar-freq', type = int, default = 1000, help='target network update frequency')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed that the dataset generated with')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
    parser.add_argument('--n-ros', type = int, nargs='+', default = [200], help='# trajectories in dataset')   

    parser.add_argument('--tau', type = float, default = 1.0, help='behavior policy')
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
        'done': dataset['done'][index],
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

    file_name = './Dataset/{}/CartPole-ep{}-tau{}-n{}-seed{}.pickle'.format(args.n_ros, args.ep_len, args.tau, args.n_ros, args.dataset_seed)

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
        norm_type = args.norm
        gamma = args.gamma

        if 'factor' not in dataset.keys():
            dataset['factor'] = np.array([gamma ** (i % args.ep_len) for i in range(dataset['obs'].shape[0])]).reshape([-1, 1])

        if norm_type == 'std_norm':
            obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
            obs_std = np.std(dataset['obs'], axis=0, keepdims=True)

            dataset['obs'] = (dataset['obs'] - obs_mean) / obs_std
            dataset['next_obs'] = (dataset['next_obs'] - obs_mean) / obs_std
            dataset['init_obs'] = (dataset['init_obs'] - obs_mean) / obs_std
            
            norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}
        else:
            norm = {'type': None, 'shift': None, 'scale': None}
            

    dqn = DQN(obs_dim, act_dim, seed=dataset_seed + seed + 2,
            q_hidden_layers=[32, 32], lr=args.lr, target_update_freq=args.tar_freq,
            ep_len=args.ep_len, gamma=args.gamma, norm=norm)            
    indicator = Indicator(obs_dim=obs_dim)
    beh_indicator = indicator.infer(dataset['obs'], dataset['acts'], dataset['factor'])
    sess.graph.finalize()

    pi_val = U.eval_policy_cartpole(env, dqn, ep_num=10, gamma=0.99, prt=True)

    log_name = 'log_seed{}_tau{}_tar{}_lr{}.pickle'.format(args.seed, args.tau, args.tar_freq, args.lr)
    logger = dqn_log_class(path='./log/{}/DQN/Dataset{}/'.format(args.n_ros, args.dataset_seed), 
                        name=log_name, env_name=env_name, beh_indicator=beh_indicator)

    q_prt_interval = 1000
    eval_interval = 2000
    eval_num = 10 

    for i in range(1, args.iter):  
        if i % args.tar_freq == 0:
            dqn.sync()

        data = sample_data(dataset, args.bs)
        error = dqn.train(data)

        if i % q_prt_interval == 0 and i != 0:
            print('-------------------------------------')
            print('Iter ', i)
            print('Bellman Error', error)
            print('Estimated Pi Val ', pi_val)      
            logger.update_error_info(i, error)
            print('-------------------------------------\n\n')
        
        if i % eval_interval == 0:
            print('Command is ', command)  
            pi_val, obs_tar, acts_tar = U.eval_policy_cartpole(env, dqn, ep_num=eval_num, gamma=args.gamma, prt=False, save_data=True)
            factor_tar = np.array([gamma ** (i % ep_len) for i in range(obs_tar.shape[0])]).reshape([-1, 1])

            # normalize data
            obs_tar = (obs_tar - obs_mean) / obs_std            
            tar_indicator = indicator.infer(obs_tar, acts_tar, factor_tar)
            logger.update_indicator_info(i, pi_val, tar_indicator)            

        if i % 10 == 0:
            logger.dump()


if __name__ == '__main__':
    args = get_parser()

    args_list = []
    for dataset_seed in args.dataset_seed:
        for seed in args.seed:
            for n_ros in args.n_ros:
                args_copy = deepcopy(args)
                args_copy.dataset_seed = dataset_seed
                args_copy.seed = seed
                args_copy.n_ros = n_ros
                args_list.append(args_copy)
    
    with Pool(processes=len(args_list), maxtasksperchild=1) as p:
        p.map(main, args_list, chunksize=1)
