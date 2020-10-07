import numpy as np
import gym
import pickle
import argparse
import time
import sys
import multiprocessing

import utils as U
from Model.CI_Opt_Class import CI_Opt
from Model.Indicator_Class import Indicator
from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from Logger.logger import opt_log_class
from multiprocessing import Pool
from copy import deepcopy


def get_parser():
    parser = argparse.ArgumentParser(description='CI-Opt')
    parser.add_argument('--iter', type = int, default = 1500, help='training iteration for CI-Opt')

    parser.add_argument('--q-iter', type = int, default = 500, help='training iteration for Q')
    parser.add_argument('--q-lr', type = float, default = 5e-3, help='learning rate for Q')
    parser.add_argument('--q-bs', type = int, default = 500, help='batch size for Q')  
    parser.add_argument('--pi-iter', type = int, default = 500, help='training iteration for pi')
    parser.add_argument('--pi-lr', type = float, default = 5e-3, help='learning rate for pi')
    parser.add_argument('--pi-bs', type = int, default = 500, help='batch size for pi')  

    parser.add_argument('--gamma', type = float, default = 0.99, help='discounted factor')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed that the dataset generated with')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
    parser.add_argument('--n-ros', type = int, nargs='+', default = [200], help='# trajectories in dataset')   

    parser.add_argument('--scale', type = float, default = 2.5, help='expecated discounted w(s,a)')
    parser.add_argument('--pi-type', type = str, choices=['upper', 'lower'], default = 1.0, help='')
    
    parser.add_argument('--norm', type = str, default = 'std_norm', help='normalization type')
    parser.add_argument('--tau', type = float, default = 0.1, help='behavior policy')
    args = parser.parse_args()

    return args

def sample_data(dataset, sample_num):
    data_size = dataset['obs'].shape[0]
    index = np.random.choice(data_size, sample_num)
    init_index = np.random.choice(dataset['init_obs'].shape[0], 50)

    return {
        'obs': dataset['obs'][index],
        'next_obs': dataset['next_obs'][index],
        'acts': dataset['acts'][index],
        'rews': dataset['rews'][index],
        'init_obs': dataset['init_obs'][init_index],
        'factor': dataset['factor'][index],
        'done': dataset['done'][index],
    }

def main(args):
    command = sys.executable + " " + " ".join(sys.argv)
    env_name = "CartPole"
    ep_len = args.ep_len
    dataset_seed = args.dataset_seed
    seed = args.seed
    U.set_seed(dataset_seed + seed)

    if args.pi_type == 'lower':
        assert args.pi_iter == 1, 'Should use a small iteration number.'

    env = CartPoleEnv(max_ep_len=ep_len, seed=seed)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    q_net = Q_network(obs_dim, act_dim, seed=100, default_tau=args.tau)
    U.initialize_all_vars()
    q_net.load_model('./CartPole_Model/Model')

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
            
    
    alg = CI_Opt(obs_dim, act_dim, seed=dataset_seed + seed + 2,
            q_hidden_layers=[32, 32], q_lr=args.q_lr, pi_type=args.pi_type, 
            pi_hidden_layers=[32, 32], pi_lr=args.pi_lr,
            scale=args.scale, ep_len=args.ep_len,
            gamma=args.gamma, norm=norm)
            
    indicator = Indicator(obs_dim=obs_dim)
    beh_indicator = indicator.infer(dataset['obs'], dataset['acts'], dataset['factor'])

    pi_val = U.eval_policy_cartpole(env, alg, ep_num=10, gamma=args.gamma, prt=True)

    log_name = 'log_seed{}_tau{}_q{}_pi{}.pickle'.format(args.seed, args.tau, args.q_iter, args.pi_iter)
    logger = opt_log_class(path='./log/{}/CI_OPT/{}/{}/Dataset{}/'.format(args.n_ros, args.pi_type, args.scale, args.dataset_seed), 
                        name=log_name, env_name=env_name, beh_indicator=beh_indicator)
    
    value_est_list = []
    pi_prt_interval = min(250, args.pi_iter)
    q_prt_interval = min(250, args.q_iter)
    
    eval_interval, eval_num = 5, 10
    ds_tv, sample_tv = 0.0, 0.0

    ds_pb_prob = q_net.get_probabilities(dataset['obs'], norm)

    def compute_ds_tv(prob_1, prob_2, discount):
        return np.mean(np.abs(prob_1 - prob_2) * discount)

    base = 0
    for iter in range(args.iter):
        print('Command is ', command)        
        for i in range(1, args.q_iter + 1):
            data_sup = sample_data(dataset, args.q_bs)
            data_inf = sample_data(dataset, args.q_bs)
            sup_q_loss, inf_q_loss = alg.train_q(data_sup, data_inf)

            if i % q_prt_interval == 0 and i != 0:
                print('-------------------------------------')
                print('Iter ', base + i)
                print('Sup Loss ', sup_q_loss, '; Inf loss ', inf_q_loss)
                print('DS_TV: ', ds_tv, '; Sample_TV :', sample_tv)
                print('Estimated Pi Val ', pi_val)
                lower_q, upper_q, lower_bound, upper_bound = alg.evaluation(dataset)
                print('True: {}. \nUpper q: {} \nLower q: {}\nUpperBound:{}\nLowerBound: {}'.format(
                        pi_val, upper_q, lower_q, upper_bound, lower_bound))
                print('-------------------------------------\n\n')
                logger.update_bound_info(base + i, lower_bound, upper_bound)
        
        base += args.q_iter                
        value_est = (lower_q + upper_q) / 2.0
        value_est_list.append(value_est)

        for i in range(1, args.pi_iter + 1):
            data = sample_data(dataset, args.pi_bs)                
            upper_bound, lower_bound = alg.train_pi(data)

            if i % pi_prt_interval == 0:
                print('-------------------------------------')
                print('Iter ', base + i)
                print('Sup Loss ', sup_q_loss, '; Inf loss ', inf_q_loss)
                print('DS_TV: ', ds_tv, '; Sample_TV :', sample_tv)
                print('Estimated Pi Val ', pi_val)
                lower_q, upper_q, lower_bound, upper_bound = alg.evaluation(dataset)
                print('True: {}. \nUpperBound:{}\nLowerBound: {}'.format(
                        pi_val, upper_bound, lower_bound))                        
                print('-------------------------------------\n\n')
                logger.update_bound_info(base + i, lower_bound, upper_bound)

        base += args.pi_iter            

        if iter % eval_interval == 0:
            pi_val, obs_tar, acts_tar = U.eval_policy_cartpole(env, alg, ep_num=eval_num, gamma=args.gamma, prt=False, save_data=True)
            factor_tar = np.array([gamma ** (i % ep_len) for i in range(obs_tar.shape[0])]).reshape([-1, 1])

            # normalized input
            obs_tar = (obs_tar - obs_mean) / obs_std

            tar_indicator = indicator.infer(obs_tar, acts_tar, factor_tar)
            logger.update_pi_info(base + i, pi_val, tar_indicator)            

            '''Compute TV Distance'''
            ds_pe_prob = alg.get_pi(dataset['obs'])
            ds_tv = compute_ds_tv(ds_pb_prob, ds_pe_prob, dataset['factor'])
            sample_pe_prob = alg.get_pi(obs_tar)
            sample_pb_prob = q_net.get_probabilities(obs_tar, norm)
            sample_tv = compute_ds_tv(sample_pe_prob, sample_pb_prob, factor_tar)

            logger.update_tv_info(iter, ds_tv=ds_tv, sample_tv=sample_tv)

        if iter % 10 == 0:
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
