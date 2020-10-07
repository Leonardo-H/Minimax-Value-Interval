import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import time


def get_parser():
    parser = argparse.ArgumentParser(description='MQL')
    parser.add_argument('--alg', type = str, nargs='+',
                                default=['CI_OPE', 'MQL_Interval'], help='alg to plot')
    parser.add_argument('--dataset', type = int, nargs='+', default=None, help='datasets to plot')
    parser.add_argument('--tau', type = float, default=1.5, help='target policy to plot')
    parser.add_argument('--type', type = str, choices=['ratio', 'length'], default=None, help='which to plot')
    parser.add_argument('--scale', type = float, default=2.5, help='scale for W function class')
    parser.add_argument('--y-lim', type = float, nargs='+', default=None, help='truncate value range')
    parser.add_argument('--dir', type = str, default='bootstrap_log', help='plot type')
    parser.add_argument('--n-ros', type = int, nargs='+', default=200, help='which n-ros to plot')
    parser.add_argument('--avg-num', type = int, default=10, help='how many to average together')
    parser.add_argument('--max-index', type = int, default=1, help='the index-max')
    parser.add_argument('--iter', type = int, default=100000, help='which interval to plot')
    
    args = parser.parse_args()

    return args

def plot_sample_size(args=None, colors=None, markers=None, legend=None):
    algs = args.alg
    tau = args.tau
    scale = args.scale    

    n_ros = args.n_ros      
    log_dir = args.dir
    max_index = args.max_index

    assert len(algs) <= len(colors)

    os.chdir(log_dir)
    with open(os.path.join('OnPolicy', str(tau), 'log.pickle'), 'rb') as f:
        on_policy = pickle.load(f)['True_Rew']

    n_ros.sort()

    alg_dict = {}
    for alg in algs:
        alg_dict[alg] = {'mean': [], 'std_error': []}
    
    for nr in n_ros:
        os.chdir(str(nr))
        for alg in algs:
            os.chdir(os.path.join(alg, str(scale)))
            datasets = os.listdir()

            plot_value = []

            for d in datasets:
                if not os.path.exists(os.path.join(d, str(tau))):
                    continue
                os.chdir(os.path.join(d, str(tau)))
                upper_bound_list = []
                lower_bound_list = []
                for log in os.listdir():
                    with open(log, 'rb') as f:
                        record = pickle.load(f)
                        upper_bound, lower_bound = [], []
                        bound_info = record['Bound_Info']
                        while bound_info[-1]['iter'] > args.iter:
                            bound_info.pop()
                        for i in range(1, args.avg_num + 1):
                            last_bound_info = bound_info[-i]
                            upper_bound.append(last_bound_info['upper_bound'])
                            lower_bound.append(last_bound_info['lower_bound'])
                        
                        upper_bound = sum(upper_bound) / len(lower_bound)
                        lower_bound = sum(lower_bound) / len(lower_bound)

                        if upper_bound > lower_bound:
                            upper_bound_list.append(upper_bound)
                            lower_bound_list.append(lower_bound)
                        else:
                            upper_bound_list.append(lower_bound)
                            lower_bound_list.append(upper_bound)

                upper_bound_list.sort()
                lower_bound_list.sort()

                if len(upper_bound_list) == 1:
                    max_upper = upper_bound_list[0]
                else:
                    max_upper = upper_bound_list[-max_index]
                if len(lower_bound_list) == 1:
                    min_lower = lower_bound_list[0]
                else:
                    min_lower = lower_bound_list[max_index-1]
                    
                if args.type == 'length':
                    plot_value.append(max_upper - min_lower)
                elif args.type == 'ratio':
                    valid = on_policy <= max_upper and on_policy >= min_lower
                    plot_value.append(valid * 1.0)
                else:
                    raise NotImplementedError

                os.chdir('../..')
            os.chdir('../..')
            
            value = np.mean(plot_value) 
            std_error = np.std(plot_value, axis=0, ddof=1) / np.sqrt(len(plot_value))  

            alg_dict[alg]['mean'].append(value)
            alg_dict[alg]['std_error'].append(std_error)

        os.chdir('..')

    labels = ['# of Trajectories=' + str(nr) for nr in n_ros]
    x_corrd = [i for i in range(len(n_ros))]
    for alg, color, marker in zip(algs, colors, markers):
        suffix = ''
        if alg == 'CI_OPE':
            prefix = r'$\rm [LB_q, UB_q]$'
        else:
            prefix = r"$\rm [LB_q' , UB_q' ]$"       
        value = alg_dict[alg]['mean']
        std_error = alg_dict[alg]['std_error']
        plt.scatter(x_corrd, value, c='', linewidths=10, edgecolors=color, marker=marker, label=prefix + ' ' + legend + ' ' + suffix, s=3000)

        for i in range(len(x_corrd) - 1):
            plt.plot([x_corrd[i], x_corrd[i+1]], [value[i], value[i+1]], color=color)

    y_label = {
        'ratio': 'Validity Ratio',
        'length': 'Interval Length',
    }

    if args.plot_title:
        plt.xlabel('# of Trajectories', fontsize=80)
        plt.ylabel(y_label[args.type], fontsize=80)
        plt.xticks(range(len(n_ros)), n_ros, fontsize=80)
        plt.yticks(fontsize=80)
        plt.ylim(args.y_lim)
        plt.title('Target Policy ' + r'$\tau=$' + str(tau), fontsize=100)

if __name__ == '__main__':
    args = get_parser()    

    plt.figure(figsize=(32, 24))
    
    
    args.dir = 'bootstrap_log'
    args.type = 'length'
    args.n_ros = [30, 40, 50, 100, 150]
    args.plot_title = False
    plot_sample_size(args, colors=['r', 'b'], markers=['d', 's'], legend='With Bootstrapping') 

    args.plot_title = True
    args.dir = 'log'
    args.n_ros = [30, 40, 50, 100, 150]
    os.chdir('..')
    plot_sample_size(args, colors=['g', 'orange'], markers=['^', 'o'], legend='Without Bootstrapping')

    plt.tight_layout()
    plt.savefig('./bootstrap_len.pdf')

    plt.figure(figsize=(32, 24))
    args.type = 'ratio'
    args.dir = 'bootstrap_log'
    args.n_ros = [30, 40, 50, 100, 150]
    os.chdir('..')
    args.plot_title = False
    plot_sample_size(args, colors=['r', 'b'], markers=['d', 's'], legend='With Bootstrapping') 

    args.plot_title = True
    args.dir = 'log'
    args.n_ros = [30, 40, 50, 100, 150]
    os.chdir('..')
    plot_sample_size(args, colors=['g', 'orange'], markers=['^', 'o'], legend='Without Bootstrapping')

    plt.legend(prop={'size': 60}, loc='lower right')
    plt.tight_layout()
    plt.savefig('./bootstrap_val.pdf')
