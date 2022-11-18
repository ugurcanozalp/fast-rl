
import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from fastrl.algos import SoftActorCritic
from fastrl.nets.heads import SquashedGaussianHead

class ActorNet(nn.Module):
	def __init__(self, n_x, n_u):
		super(ActorNet, self).__init__()
		self.activation = nn.GELU()
		self.fc1 = nn.Sequential(nn.Linear(n_x[0], 256), nn.LayerNorm(256, elementwise_affine=False))
		self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256, elementwise_affine=False))
		self.fc3 = nn.Linear(256, 2*n_u[0])
		self.head = SquashedGaussianHead(n_u[0])

	def forward(self, x):
		f = x
		f = self.fc1(f)
		f = self.activation(f)
		f = self.fc2(f)
		f = self.activation(f)
		f = self.fc3(f)
		return self.head(f)

class CriticNet(nn.Module):
	def __init__(self, n_x, n_u):
		super(CriticNet, self).__init__()
		self.activation = nn.GELU()
		self.fc1 = nn.Sequential(nn.Linear(n_x[0] + n_u[0], 256))
		self.fc2 = nn.Sequential(nn.Linear(256, 256))
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x, u):
		f = torch.concat([x, u], dim=-1)
		f = self.fc1(f)
		f = self.activation(f)
		f = self.fc2(f)
		f = self.activation(f)
		f = self.fc3(f)
		return f
		
env = gym.make("Ant-v4", terminate_when_unhealthy=False, render_mode="human")

optimizer_args = {
	"lr": 3e-4,
}

agent = SoftActorCritic(env, ActorNet, CriticNet, optimizer_args, buffer_size=int(1e6), device="cuda")

agent.train(max_num_episodes=200, test_interval=50, warmup_episodes=1)
