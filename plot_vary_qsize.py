import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='MQL')
    parser.add_argument('--dataset', type = int, nargs='+', default=None, help='datasets for plot')
    parser.add_argument('--tau', type = float, default=2.5, help='tau for target policy')
    parser.add_argument('--scale', type = float, default=2.5, help='scale for W function class')
    parser.add_argument('--x-lim', type = float, nargs='+', default=[0, 300000], help='truncate steps')
    parser.add_argument('--y-lim', type = float, nargs='+', default=None, help='truncate value range')
    parser.add_argument('--dir', type = str, default='log', help='directory of log files')
    parser.add_argument('--n-ros', type = int, default=200, help='which n-ros to plot')    
    parser.add_argument('--save-path', type = str, default='./CI_Vary_Qsize.png', help='path to save picture')
    
    args = parser.parse_args()

    return args

def plot_tau(args=None):
    tau = args.tau
    scale = args.scale

    n_ros = args.n_ros   
    log_dir = args.dir

    plt.figure(figsize=(32, 24))

    os.chdir(log_dir)
    # with open(os.path.join('OnPolicy', str(tau), 'log.pickle'), 'rb') as f:
    #     on_policy = pickle.load(f)['True_Rew']
    on_policy = 280
    index = 0
    x_corrd = []
    
    os.chdir(str(n_ros)) 
    alg = 'CI_OPE'
    for pair in args.pairs:
        index += 1
        
        os.chdir(os.path.join(alg, str(scale)))
        
        if args.dataset is None:
            datasets = os.listdir()
        else:
            datasets = ['Dataset' + str(i) for i in args.dataset]
        
        x, upper, lower = [], [], []

        length_set = []

        for d in datasets:
            if not os.path.exists(os.path.join(d, str(tau))):
                continue
                
            os.chdir(os.path.join(d, str(tau)))
            for log in os.listdir():
                if 'qs{}_ws{}'.format('x'.join(pair['qs']), 'x'.join(pair['ws'])) not in log:
                    continue
                with open(log, 'rb') as f:
                    record = pickle.load(f)
                    x_, upper_, lower_ = plot_with_record(record, args)
                    x.append(x_)
                    upper.append(upper_)
                    lower.append(lower_)
                    length_set.append(x_.size)
            os.chdir('../..')                       
        
        minimum = min(length_set)
        for i in range(len(x)):
            x[i] = x[i][:, 0:minimum]
            lower[i] = lower[i][:, 0:minimum]
            upper[i] = upper[i][:, 0:minimum]
            
        upper = np.concatenate(upper, axis=0)[:, -10:]
        lower = np.concatenate(lower, axis=0)[:, -10:]
        
        upper_avg = np.mean(upper, axis=1)
        lower_avg = np.mean(lower, axis=1)

        upper_ste = np.std(upper_avg, axis=0, ddof=1) / np.sqrt(upper_avg.shape[0])
        lower_ste = np.std(lower_avg, axis=0, ddof=1) / np.sqrt(lower_avg.shape[0])

        x_corrd.append('x'.join(pair['qs']))
        
        upper, lower = np.mean(upper), np.mean(lower)
        shift = 0.5

        lower_clr = 'b'
        plt.fill_between([index - shift, index + shift], [upper - 2 * upper_ste] * 2, [upper + 2 * upper_ste] * 2, alpha=0.25, facecolor='r')
        plt.fill_between([index - shift, index + shift], [lower - 2 * lower_ste] * 2, [lower + 2 * lower_ste] * 2, alpha=0.25, facecolor=lower_clr)
        
        plt.plot([index, index], [upper, lower], color='black', linewidth=10.0)
        
        if len(x_corrd) == len(pairs):
            plt.plot([index - shift, index + shift], [upper, upper], '-', color='r', linewidth=10.0, label=r'${\rm UB}_q$' + r'$(\approx{\rm LB}_w$)')
            plt.plot([index - shift, index + shift], [lower, lower], ':', color=lower_clr, linewidth=10.0, label=r'${\rm LB}_q$' + r'$(\approx{\rm UB}_w$)')
        else:
            plt.plot([index - shift, index + shift], [upper, upper], '-', color='r', linewidth=10.0)
            plt.plot([index - shift, index + shift], [lower, lower], ':', color=lower_clr, linewidth=10.0)
        
        os.chdir('../..')
    os.chdir('../..')


    plt.plot([1-0.5, index+0.5], [on_policy, on_policy], color='purple', linewidth=3.0, label='Groundtruth')

    plt.xticks([i for i in range(1, index + 1)], x_corrd)
    plt.xlabel('Q Network Size', fontsize=80)
    plt.ylabel('Lower/Upper Bound', fontsize=80)
    
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    

    if args.y_lim is not None:
        plt.ylim(args.y_lim)

    plt.legend(loc='lower left', prop={'size': 60})

    plt.title('Target Policy ' + r'$\tau=$' + str(tau), fontsize=100)

    plt.tight_layout()
    plt.savefig(args.save_path)


def plot_with_record(record, args):
    x_lim = args.x_lim

    bound_info = record['Bound_Info']
    all_x = []
    all_upper = []
    all_lower = []
    
    iter = x_lim[0]

    for index in range(1, len(bound_info)):
        n_iter, n_lower, n_upper = bound_info[index]['iter'], bound_info[index]['lower_bound'], bound_info[index]['upper_bound']

        if iter >= x_lim[0] and iter <= x_lim[1]:
            all_x.append(n_iter)
            all_upper.append(n_upper)
            all_lower.append(n_lower)
        iter = n_iter
        
    return np.array([all_x]), np.array([all_upper]), np.array([all_lower])


if __name__ == '__main__':
    args = get_parser()
    qs = ['32']

    pairs = [
        {'qs': ['32'], 'ws':['1']},
        {'qs': ['32'], 'ws':['3']},
        {'qs': ['32'], 'ws':['10']},
        {'qs': ['32'], 'ws':['32']},
        {'qs': ['32'], 'ws':['32', '32']},
    ]

    for d in pairs:
        d['qs'] = qs
    
    args.pairs = pairs
    plot_tau(args)
