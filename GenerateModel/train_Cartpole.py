import gym
import sys
import argparse

sys.path.append('..')
sys.path.append('.')
sys.path.append('../..')
from baselines import deepq
from baselines.deepq.CartPole import CartPoleEnv


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main(args):    
    env_name = "CartPole-v0"
    
    env = CartPoleEnv(max_ep_len=args.ep_len, seed=args.seed, append=False)
    model = deepq.models.mlp([64])
    max_timesteps = args.steps
    act = deepq.learn(
        env,
        env_name=env_name,
        q_func=model,
        lr=args.lr,
        max_timesteps=max_timesteps,
        buffer_size=50000,
        exploration_fraction=0.0001,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback,
        eval=False,
    )
    # print("Saving model to cartpole_model.pkl")
    # act.save(env_name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MQL_SGDA')
    parser.add_argument('--steps', type = int, default = 1000000, help='total number of steps')
    parser.add_argument('--lr', type = float, default = 1e-3, help='learning rate')
    parser.add_argument('--ep-len', type = int, default = 1500, help='length of episodes')
    parser.add_argument('--seed', type = int, default = 100, help='random seed')
    
    args = parser.parse_args()

    main(args)
