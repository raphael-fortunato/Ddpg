import gym
import os
import numpy as np
import torch
import random
from ddpg_agent import Agent
from buffer import ReplayBuffer
from models import Actor, Critic
from CustomTensorBoard import ModifiedTensorBoard
from arguments import GetArgs


def get_params(env):
        obs = env.reset()
        params = {'observation': obs.shape[0],
                        'action': env.action_space.shape[0],
                        'max_action': env.action_space.high[0],
                        }
        params['max_timesteps'] = env._max_episode_steps
        return params

if __name__ == '__main__':
        # take the configuration for the HER
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['IN_MPI'] = '1'
        args = GetArgs()
        env = gym.make('Pendulum-v0')
        env_param = get_params(env)

        #set seeds
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)
        env.seed(0)

        agent = Agent(env, env_param,args)
        agent.Explore()
