import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Actor(nn.Module):
	def __init__(self,env_params):
		super(Actor, self).__init__() 
		self.max_action = env_params['max_action']
		
		self.fc1 = nn.Linear(env_params['observation'],256)
		self.batch_norm1 = nn.BatchNorm1d(256)
		
		self.fc2 = nn.Linear(256, 256)
		self.batch_norm2 = nn.BatchNorm1d(256)
	
		self.fc3 = nn.Linear(256, 256)
		self.batch_norm3 = nn.BatchNorm1d(256)

		self.action_out = nn.Linear(256, env_params['action'])


		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1,f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2,f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc3.weight.data, -f3,f3)
		torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

	def forward(self, x):
		if type(x) is not torch.Tensor:
			x = torch.tensor(x)

		x = self.fc1(x)
		x = self.batch_norm1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.batch_norm2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = self.batch_norm3(x)
		x = F.relu(x)
		action = self.max_action * torch.tanh(self.action_out(x))
		return action


class Critic(nn.Module):
	def __init__(self ,env_params):
		super(Critic, self).__init__()
		self.max_action = env_params['max_action']
		self.fc1 = nn.Linear(env_params['observation'] + env_params['action'],256 )
		self.batch_norm1 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 256)
		self.batch_norm2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 256)
		self.batch_norm3 = nn.BatchNorm1d(256)
		self.q_out = nn.Linear(256, 1)


		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1,f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2,f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc3.weight.data, -f3,f3)
		torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)        

	def forward(self, x, action):
		x = torch.cat([x, action/ self.max_action], dim=1)
		x = self.fc1(x)
		x = self.batch_norm1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.batch_norm2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = self.batch_norm3(x)
		x = F.relu(x)
		q_value = self.q_out(x)
		return q_value