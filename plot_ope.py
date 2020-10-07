import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='MQL')
    parser.add_argument('--alg', type = str, nargs='+',
                                default=['CI_OPE', 'MQL_Interval'], help='alg to plot')
    parser.add_argument('--dataset', type = int, nargs='+', default=None, help='dataset to plot')
    parser.add_argument('--tau', type = float, default=0.5, help='target policy to plot')
    parser.add_argument('--scale', type = float, default=2.5, help='scale for W function class')
    parser.add_argument('--x-lim', type = float, nargs='+', default=[0, 300000], help='truncate steps')
    parser.add_argument('--y-lim', type = float, nargs='+', default=None, help='truncate value range')
    parser.add_argument('--dir', type = str, default='log', help='plot type')
    parser.add_argument('--avg', type=int, default=None, help='whether to average the data across iteration')
    parser.add_argument('--n-ros', type = int, nargs='+', default=[200], help='which n-ros to plot')
    parser.add_argument('--save-path', type = str, default='./OPE.png', help='path to save picture')
    
    args = parser.parse_args()

    return args

def plot_tau(args=None):    
    algs = args.alg
    tau = args.tau
    scale = args.scale

    x_lim = args.x_lim        
    log_dir = args.dir

    legend = {
        'CI_OPE_Upper': r'${\rm UB}_q$',
        'CI_OPE_Lower': r'${\rm LB}_q$',
        'MQL_Interval_Upper': r'${\rm UB}_q$' + '\'',
        'MQL_Interval_Lower': r'${\rm LB}_q$' + '\'',
    }

    colors = ['r', 'orange', 'gold', 'g', 'b', 'purple']
    assert len(algs) <= len(colors)

    plt.figure(figsize=(32, 24))

    os.chdir(log_dir)
    with open(os.path.join('OnPolicy', str(tau), 'log.pickle'), 'rb') as f:
        on_policy = pickle.load(f)['True_Rew']

    for n_ros in args.n_ros:
        os.chdir(str(n_ros)) 
        for alg in algs:
            color = colors.pop(0)   
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
                    with open(log, 'rb') as f:
                        record = pickle.load(f)
                        x_, upper_, lower_ = get_data_from_record(record, args)
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
            x_mean = np.mean(np.concatenate(x, axis=0), axis=0)
            upper = np.concatenate(upper, axis=0)
            lower = np.concatenate(lower, axis=0)            

            upper_mean = np.mean(upper, axis=0)
            lower_mean = np.mean(lower, axis=0)

            upper_ste = np.std(upper, axis=0, ddof=1) / np.sqrt(upper.shape[0])
            lower_ste = np.std(lower, axis=0, ddof=1) / np.sqrt(lower.shape[0])

            plt.plot(x_mean, upper_mean, color=color, label=legend[alg + '_Upper'], linewidth=10.0)
            plt.fill_between(x_mean, upper_mean - 2 * upper_ste, upper_mean + 2 * upper_ste, alpha=0.25, facecolor=color)

            plt.plot(x_mean, lower_mean, ':', color=color, label=legend[alg + '_Lower'], linewidth=10.0)
            plt.fill_between(x_mean, lower_mean - 2 * lower_ste, lower_mean + 2 * lower_ste, alpha=0.25, facecolor=color)


            os.chdir('../..')
        os.chdir('..')
    os.chdir('..')

    plt.plot([x_lim[0] - 1000, x_lim[1] + 1000], [on_policy, on_policy], color='purple', linewidth=10.0, label='Groundtruth')

    plt.xticks([0, 100000, 200000, 300000], [0, 1, 2, 3])
    plt.xlabel('Iteration (1e5)', fontsize=80)
    plt.ylabel('Lower/Upper Bound', fontsize=80)
    plt.legend(prop={'size': 60}, loc='lower right', framealpha=1.0, ncol = 3)
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)

    if args.y_lim is not None:
        plt.ylim(args.y_lim)

    plt.title('Target Policy ' + r'$\tau=$' + str(tau), fontsize=100)
   
    plt.savefig(args.save_path)

def get_average(array, d=100):
    new_array = []
    for i in range(len(array)):
        start = max(i - d, 0)
        end = min(i + d, len(array))
        new_array.append(np.mean(array[start:end]))
    return new_array

def get_data_from_record(record, args):
    x_lim = args.x_lim

    bound_info = record['Bound_Info']
    iter, lower, upper = bound_info[0]['iter'], bound_info[0]['lower_bound'], bound_info[0]['upper_bound']

    all_x = []
    all_upper = []
    all_lower = []
    
    for index in range(1, len(bound_info)):
        n_iter, n_lower, n_upper = bound_info[index]['iter'], bound_info[index]['lower_bound'], bound_info[index]['upper_bound']

        if iter >= x_lim[0] and iter <= x_lim[1]:
            all_x.append(n_iter)
            all_upper.append(n_upper)
            all_lower.append(n_lower)
            last_iter, lower, upper = n_iter, n_lower, n_upper
        iter = n_iter

    if args.avg:
        all_upper = get_average(all_upper, args.avg)
        all_lower = get_average(all_lower, args.avg)

    return np.array([all_x]), np.array([all_upper]), np.array([all_lower])


if __name__ == '__main__':
    args = get_parser()
    
    plot_tau(args)
