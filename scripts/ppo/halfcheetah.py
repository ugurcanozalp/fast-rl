
import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from fastrl.algos import ProximalPolicyOptimization
from fastrl.nets.heads import SquashedGaussianHead

class ActorNet(nn.Module):
	def __init__(self, n_x, n_u):
		super(ActorNet, self).__init__()
		self.activation = nn.GELU()
		self.fc1 = nn.Sequential(nn.Linear(n_x[0], 256), nn.LayerNorm(256))
		self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128))
		self.fc3 = nn.Linear(128, 2*n_u[0])
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
	def __init__(self, n_x):
		super(CriticNet, self).__init__()
		self.activation = nn.GELU()
		self.fc1 = nn.Sequential(nn.Linear(n_x[0], 256), nn.LayerNorm(256))
		self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		f = self.fc1(x)
		f = self.activation(f)
		f = self.fc2(f)
		f = self.activation(f)
		f = self.fc3(f)
		return f

env = gym.make("HalfCheetah-v4", render_mode="human")

optimizer_args = {
	"lr": 3e-4,
}

agent = ProximalPolicyOptimization(env, ActorNet, CriticNet, optimizer_args, buffer_size=int(1e6), device="cuda")

agent.train(max_num_episodes=200, test_interval=50, warmup_episodes=1)
