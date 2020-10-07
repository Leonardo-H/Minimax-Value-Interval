import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import time

def get_parser():
    parser = argparse.ArgumentParser(description='MQL')
    parser.add_argument('--alg', type = str, nargs='+',
                                default=['CI_Opt_Upper', 'CI_Opt_Lower', 'DQN'], help='alg to plot')
    parser.add_argument('--dataset', type = int, nargs='+', default=None, help='dataset to plot')
    parser.add_argument('--scale', type = float, default=0.3, help='scale for W function class')
    parser.add_argument('--x-lim', type = float, nargs='+', default=[0, 10000], help='truncate steps')
    parser.add_argument('--y-lim', type = float, nargs='+', default=None, help='truncate value range')
    parser.add_argument('--dir', type = str, default='log', help='directory for log')
    parser.add_argument('--n-ros', type = int, default=200, help='which n-ros to plot')  
   
    parser.add_argument('--tau', type = float, default=None, help='tau of behavior policy')
    parser.add_argument('--key', type = str, default='Pi_Val', help='plot what')
    parser.add_argument('--plot-true', action='store_true', help='whether to plot true value of behavior policy')

    parser.add_argument('--save-path', type = str, default='./Fig.png', help='directory for log')

    args = parser.parse_args()

    return args

def get_average(x, length=10):
    for i in range(len(x)):
        start = max(0, i - length)
        end = min(len(x), i + length)
        x[i] = np.mean(x[start:end])
    return x

def read_log(log, key):
    with open(log, 'rb') as f:
        record = pickle.load(f)
        
        if key == 'Pi_Val':                
            val = get_average(record[key], 10)
        else:
            tar_ind = record['Tar_Indicator']
            beh_ind = record['Beh_Indicator']
            val = []
            for i in range(len(tar_ind)):
                val.append(np.max(np.abs(tar_ind[i]-beh_ind)))
            val = get_average(val, 10)

    return val, record


def main(args=None):    
    algs = args.alg
    scale = args.scale

    n_ros = args.n_ros
    x_lim = args.x_lim        
    log_dir = args.dir
    dataset_names = ['Dataset' + str(num) for num in args.dataset]

    plot_data = {}

    colors = ['r', 'orange', 'b', 'g', 'cyan', 'purple', 'gold']
    assert len(algs) <= len(colors)

    plt.figure(figsize=(32, 24))

    os.chdir(log_dir)

    with open(os.path.join('OnPolicy', str(args.tau), 'log.pickle'), 'rb') as f:
        on_policy = pickle.load(f)['True_Rew']
    if args.plot_true:
        plt.plot([x_lim[0] - 1000, x_lim[1] + 1000], [on_policy, on_policy], color='purple', linewidth=10.0, label='Return of behavior policy')

    os.chdir(str(n_ros)) 
    
    for alg in algs: 
        if alg == 'DQN':
            os.chdir('DQN')
        elif alg == 'CI_Opt_Lower':
            os.chdir(os.path.join('CI_OPT', 'lower', str(scale)))
        elif alg == 'CI_Opt_Upper':
            os.chdir(os.path.join('CI_OPT', 'upper', str(scale)))
        else:
            raise NotImplementedError
        datasets = os.listdir()

        all_val = []
        
        for d in datasets:
            if not d in dataset_names:
                continue
            os.chdir(os.path.join(d))
            for log in os.listdir():
                if not 'log' in log:
                    continue
                if not 'tau{}'.format(args.tau) in log:
                    continue                        
                
                val, record = read_log(log, args.key)
                all_val.append(np.array(val))                                        
            os.chdir('..')

        key = args.key
        assert len(all_val) > 0, 'No data to plot'        
        
        min_len = all_val[0].shape[0]
        for val in all_val:
            if val.shape[0] < min_len:
                min_len = val.shape[0]            
        for i in range(len(all_val)):
            all_val[i] = all_val[i][:min_len].reshape([1, -1])

        all_val_concat = np.concatenate(all_val, axis=0)
        y_val = np.mean(all_val_concat, axis=0)
        if key == 'Pi_Val' or key == 'IPM':
            x_val = record['Pi_Iter'][:min_len]
        else:
            x_val = record['Iter'][:min_len]

        ste = np.std(all_val_concat, axis=0, ddof=1) / np.sqrt(all_val_concat.shape[0])

        if alg == 'DQN':
            label = 'DQN'
        else:
            if 'upper' in alg:
                label = 'MUB-PO'
            else:
                label = 'MLB-PO'
            
        plot_data[alg] = {
            'alg': alg,
            'label': label,
            'x_val': x_val,
            'y_val': y_val,
            'ste': ste,
            'all_val': all_val_concat,
        }

        if alg == 'DQN':
            os.chdir('..')
        else:
            os.chdir('../..')

        os.chdir('..')

    plot_with_data(plot_data, colors, alg, key)

def plot_with_data(plot_data, colors, alg, key):        
    for color, k in zip(colors, plot_data.keys()):
        pd = plot_data[k]
        
        x_val = pd['x_val']
        y_val = pd['y_val']
        label = pd['label']        
        
        ste = pd['ste']
        
        if key == 'Pi_Val':
            plt.plot(x_val, y_val, '-', color=color, label=label, linewidth=10.0)
            plt.fill_between(x_val, y_val - 2 * ste, y_val + 2 * ste, alpha=0.25, facecolor=color)
        else:
            plt.plot(x_val, y_val, '--', color=color, label=label, linewidth=10.0)
            plt.fill_between(x_val, y_val - 2 * ste, y_val + 2 * ste, alpha=0.25, facecolor=color)


    if args.key == 'Pi_Val':
        plt.ylabel('Discounted return', fontsize=80)
    elif args.key == 'IPM':
        plt.ylabel('IPM', fontsize=80)
    else:
        plt.ylabel(args.key, fontsize=80)
        

    plt.legend(prop={'size': 60}, loc='lower right', framealpha=1.0, bbox_to_anchor=(0.5, 0.1, 0.5, 0.5))
    plt.xticks([100000, 200000, 300000, 400000, 500000], [1, 2, 3, 4, 5], fontsize=80)
    plt.yticks(fontsize=80)   
    plt.xlabel('Iteration (1e5)', fontsize=80)
    plt.xlim(args.x_lim)
    if args.y_lim is not None:
        plt.ylim(args.y_lim)


    if args.tau == 1.0:
        plt.title('Behavior Policy Tau=1.0', fontsize=100)
    elif args.tau == 0.1:
        plt.title('Behavior Policy Tau=0.1', fontsize=100)

    plt.tight_layout()
    os.chdir('..')
    plt.savefig(args.save_path)

if __name__ == '__main__':
    args = get_parser()    
    main(args)
