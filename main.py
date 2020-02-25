import gym
import os
from mpi4py import MPI
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
	print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
	args = GetArgs()
	env = gym.make('LunarLanderContinuous-v2') 
	env_param = get_params(env)
	device = 'cuda'
	agent = Agent(env, env_param,args, device)
	agent.Explore()