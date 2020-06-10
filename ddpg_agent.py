import os
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pdb
import numpy as np
import random
from collections import deque
from tqdm import tqdm

import torch
import torch.nn.functional as F

from CustomTensorBoard import ModifiedTensorBoard


import time
from copy import deepcopy
from mpi4py import MPI
from mpi_utils import sync_networks, sync_grads

from normalizer import Normalizer
from models import Actor, Critic
from buffer import ReplayBuffer
from arguments import GetArgs
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.


class Agent:
    def __init__(self,env, env_params, args,device, models=None):
        self.env= env
        self.env_params = env_params
        self.args = args
        self.device = device

        # networks
        if models == None:
                self.actor = Actor(self.env_params).double()
                self.critic = Critic(self.env_params).double()
        else:
                self.actor , self.critic = self.LoadModels()
        # target networks used to predict env actions with
        self.actor_target = Actor(self.env_params,).double()
        self.critic_target = Critic(self.env_params).double()

        sync_networks(self.actor)
        sync_networks(self.critic)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        if(device == 'cuda'):
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()


        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.normalize = Normalizer(env_params,self.args.gamma)
        self.buffer = ReplayBuffer(1_000_000, self.env_params)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.tensorboard = ModifiedTensorBoard(log_dir = f"logs")

    def ModelsEval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def ModelsTrain(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def GreedyAction(self, state):
        self.ModelsEval()
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.double).unsqueeze(dim=0)
            action = self.actor.forward(state).detach().cpu().numpy()
        return action.squeeze()

    def NoiseAction(self, state):
        self.ModelsEval()
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.double).unsqueeze(dim=0)
            action = self.actor.forward(state).detach().cpu().numpy()
            action += self.args.noise_eps * self.env_params['max_action'] * np.random.randn(*action.shape)
            action = np.clip(action, -self.env_params['max_action'], self.env_params['max_action'])
        return action.squeeze()

    def Update(self):
        self.ModelsTrain()
        for i in range(self.args.n_batch):
            state, a_batch, r_batch, nextstate, d_batch = self.buffer.SampleBuffer(self.args.batch_size)
            a_batch = torch.tensor(a_batch,dtype=torch.double)
            r_batch = torch.tensor(r_batch,dtype=torch.double)
            # d_batch = torch.tensor(d_batch,dtype=torch.double)
            state = torch.tensor(state,dtype=torch.double)
            nextstate = torch.tensor(nextstate,dtype=torch.double)
            # d_batch = 1 - d_batch

            if self.device == 'cuda':
                a_batch = a_batch.cuda()
                r_batch = r_batch.cuda()
                # d_batch = d_batch.cuda()
                state = state.cuda()
                nextstate = nextstate.cuda()

            with torch.no_grad():
                action_next = self.actor_target.forward(nextstate)
                q_next = self.critic_target.forward(nextstate,action_next)
                q_next = q_next.detach().squeeze()
                q_target = r_batch + self.args.gamma * q_next
                q_target = q_target.detach().squeeze()

            q_prime = self.critic.forward(state, a_batch).squeeze()
            critic_loss = F.mse_loss(q_target, q_prime)

            action = self.actor.forward(state)
            actor_loss = -self.critic.forward(state, action).mean()
            # params = torch.cat([x.view(-1) for x in self.actor.parameters()])
            # l2_reg = self.args.l2_norm *torch.norm(params,2)
            # actor_loss += l2_reg

            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            sync_grads(self.critic)
            self.critic_optim.step()

        self.SoftUpdateTarget(self.critic, self.critic_target)
        self.SoftUpdateTarget(self.actor, self.actor_target)

    def Explore(self):
        for epoch in range(self.args.n_epochs +1):
            start_time = time.process_time()
            for cycle in range(self.args.n_cycles):
                for _ in range(self.args.num_rollouts_per_mpi):
                    state = self.env.reset()
                    for t in range(self.env_params['max_timesteps']):
                        action = self.NoiseAction(state)
                        nextstate, reward, done, info = self.env.step(action)
                        reward = self.normalize.normalize_reward(reward)
                        self.buffer.StoreTransition(state, action, reward, nextstate, done)
                        state = nextstate
                    self.Update()
            avg_reward = self.Evaluate()
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.tensorboard.step = epoch
                elapsed_time = time.process_time() - start_time
                print(f"Epoch {epoch} of total of {self.args.n_epochs +1} epochs, average reward is: {avg_reward}.\
                        Elapsedtime: {int(elapsed_time /60)} minutes {int(elapsed_time %60)} seconds")
                if epoch % 5 or epoch + 1 == self.args.n_epochs:
                    self.SaveModels(epoch)
            self.record(epoch)


    def Evaluate(self):
        self.ModelsEval()
        total_reward = []
        episode_reward = 0
        succes_rate = []
        for episode in range(self.args.n_evaluate):
            state = self.env.reset()
            episode_reward = 0
            for t in range(self.env_params['max_timesteps']):
                action = self.GreedyAction(state)
                nextstate, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = nextstate
                if done or t + 1 == self.env_params['max_timesteps']:
                    total_reward.append(episode_reward)
                    episode_reward = 0

        average_reward = sum(total_reward)/len(total_reward)
        min_reward = min(total_reward)
        max_reward = max(total_reward)
        global_avg_reward = MPI.COMM_WORLD.allreduce(average_reward.item(), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        global_max_reward = MPI.COMM_WORLD.allreduce(average_reward.item(), op=MPI.MAX)
        global_min_reward = MPI.COMM_WORLD.allreduce(average_reward.item(), op=MPI.MIN)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.tensorboard.update_stats(reward_avg=global_avg_reward, reward_min=global_min_reward, reward_max=global_max_reward)
        return global_avg_reward

    def record(self, epoch):
        self.ModelsEval()
        try:
            if not os.path.exists("videos"):
                os.mkdir('videos')
            recorder = VideoRecorder(self.env, path=f'videos/epoch-{epoch}-num-{MPI.COMM_WORLD.Get_rank()}.mp4')
            for _ in range(self.args.n_record):
                done =False
                state = self.env.reset()
                while not done:
                    recorder.capture_frame()
                    action = self.GreedyAction(state)
                    nextstate,reward,done,info = self.env.step(action)
                    state = nextstate
                recorder.close()
        except Exception as e:
            print(e)

    def SaveModels(self, ep):
        if not os.path.exists("models"):
            os.mkdir('models')
        torch.save(self.actor.state_dict(), os.path.join('models', 'Actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join('models', 'Critic.pt'))

    def LoadModels(self, actorpath, criticpath):
        actor = Actor(self.env_params, self.hidden_neurons)
        critic  = Critic(self.env_params, self.hidden_neurons)
        actor.load_state_dict(torch.load(actorpath))
        critic.load_state_dict(torch.load(criticpath))
        return actor, critic

    def SoftUpdateTarget(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

