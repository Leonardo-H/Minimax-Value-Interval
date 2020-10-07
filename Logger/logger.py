import numpy as np
import os
import pickle
import time

class ope_log_class():
    def __init__(self, path='.', name=None, tau=None, env_name=None, value_true=None):
        assert name is not None, 'log should have a name'
        self.dir_path = path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.path = os.path.join(self.dir_path, name)
        
        self.iter = []
        self.true_rew = []
        self.bound_info = []

        assert tau is not None
        assert env_name is not None

        self.doc = {
            'Iter': self.iter,
            'tau': tau,
            'True_Rew': value_true,
            'env_name': env_name,
            'Bound_Info': self.bound_info,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def update_bound_info(self, iter, lower_bound, upper_bound):
        self.bound_info.append({
            'iter': iter,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        })

    def info(self, string):
        print(string)


class opt_log_class():
    def __init__(self, path='.', name=None, env_name=None, beh_indicator=None):
        assert name is not None, 'log should have a name'
        self.dir_path = path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.path = os.path.join(self.dir_path, name)
        
        self.bound_info = []

        self.pi_iter = []
        self.pi_val = []
        self.tar_indicator = []

        self.tv_iter = []
        self.ds_tv = []
        self.sample_tv = []

        assert env_name is not None


        self.doc = {
            'env_name': env_name,
            'Bound_Info': self.bound_info,

            'Pi_Iter': self.pi_iter,
            'Pi_Val': self.pi_val,
            'Beh_Indicator': beh_indicator,
            'Tar_Indicator': self.tar_indicator,

            'TV_Iter': self.tv_iter,
            'DS_TV': self.ds_tv,
            'Sample_TV': self.sample_tv,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def update_tv_info(self, iter, ds_tv, sample_tv):
        self.tv_iter.append(iter)
        self.ds_tv.append(ds_tv)
        self.sample_tv.append(sample_tv)

    def update_bound_info(self, iter, lower_bound, upper_bound):
        self.bound_info.append({
            'iter': iter,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        })

    def info(self, string):
        print(string)

    def update_pi_info(self, iter, pi_val, tar_indicator):
        self.pi_iter.append(iter)
        self.pi_val.append(pi_val)
        self.tar_indicator.append(tar_indicator)


class dqn_log_class():
    def __init__(self, path='.', name=None, env_name=None, beh_indicator=None):
        assert name is not None, 'log should have a name'
        self.dir_path = path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.path = os.path.join(self.dir_path, name)
        
        self.iter = []
        self.bellman_error = []

        self.tar_indicator = []
        self.max_diff = []

        self.pi_val = []
        self.pi_iter = []

        self.doc = {
            'Iter': self.iter,
            'env_name': env_name,
            'Beh_Indicator': beh_indicator,
            'Tar_Indicator': self.tar_indicator,
            'Pi_Iter': self.pi_iter,
            'Pi_Val': self.pi_val,
            'Bellman_Error': self.bellman_error,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def update_error_info(self, iter, error):
        self.iter.append(iter)
        self.bellman_error.append(error)

    def update_indicator_info(self, iter, pi_val, tar_indicator=None):
        self.pi_iter.append(iter)
        self.pi_val.append(pi_val)
        self.tar_indicator.append(tar_indicator)
        