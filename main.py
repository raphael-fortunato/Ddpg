import gym
import os
from mpi4py import MPI
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
        params['max_timesteps'] = 200
        return params

if __name__ == '__main__':
        # take the configuration for the HER
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['IN_MPI'] = '1'
        args = GetArgs()
        env = gym.make('BipedalWalker-v2')
        env_param = get_params(env)


        #setting seeds
        env.seed(0 + MPI.COMM_WORLD.Get_rank())
        np.random.seed(0 + MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(0 + MPI.COMM_WORLD.Get_rank())
        random.seed(0 + MPI.COMM_WORLD.Get_rank())

        device = 'cpu'
        print(f"Process {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}")
        agent = Agent(env, env_param,args, device)
        agent.Explore()
