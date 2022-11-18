
import time
import os
import random
import math
import torch
from torch import nn
from torch.distributions.kl import kl_divergence

from fastrl.agent import Agent


class ProximalPolicyOptimization(Agent):
	
	_agent_name = "ppo"

	def __init__(self, env, actor_nn, critic_nn, optimizer_args, 
			gamma=0.99, alpha=0.05, eps=0.2, epochs=10, 
			**kwargs):
		super(ProximalPolicyOptimization, self).__init__(env, **kwargs)
		self._env = env
		self._alpha = alpha
		self._gamma = gamma
		self._eps = eps
		self._critic_loss_fcn = nn.MSELoss()
		self._epochs = epochs
		#
		self._actor = actor_nn(self._nx, self._nu).to(self._device)
		self._old_actor = actor_nn(self._nx, self._nu).to(self._device)
		self._hard_update(self._actor, self._old_actor)
		self._critic = critic_nn(self._nx).to(self._device)
		#
		self._actor_optim = torch.optim.Adam(self._actor.parameters(), **optimizer_args)
		self._critic_optim = torch.optim.Adam(self._critic.parameters(), **optimizer_args)

	def learn_on_step(self):
		pass

	def learn_on_episode(self):
		if self._experience_memory.size < self._t:
			return None
		x, u, r, xp, done = self._experience_memory.sample_last_k(self._t) # sample last episode
		for e in range(self._epochs):
			# construct advantage estimates
			returns = torch.zeros_like(r) # return-to-go
			with torch.no_grad():
				returns[-1] = r[-1] + self._gamma * done[-1] * self._critic(xp[-1:])[0]
			for i in reversed(range(self._t-1)):
				returns[i] = r[i] + self._gamma * returns[i+1]
			# critic learning
			v_err = returns.unsqueeze(-1) - self._critic(x) # advantage estimate
			v_loss = 0.5*v_err.pow(2).mean()
			self._critic_optim.zero_grad()
			v_loss.backward()
			# policy learning
			with torch.no_grad(): # for kl target
				_, u_pred_dist_old = self._old_actor(x)
				log_probs_old = u_pred_dist_old.log_prob(u)
				adv = returns.unsqueeze(-1) - self._critic(x) # advantage estimate
				adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
			u_pred, u_pred_dist = self._actor(x)
			e_pred = - u_pred_dist.log_prob(u_pred).sum(dim=-1, keepdim=True) # entropy
			# Calculate surrogate
			log_probs = u_pred_dist.log_prob(u)
			likelihood_ratio = ( log_probs - log_probs_old ).exp()
			surrogate_default = adv_norm * likelihood_ratio
			# Clipping the constraint
			likelihood_ratio_clip = torch.clamp(likelihood_ratio, 1 - self._eps, 1 + self._eps)
			# Calculate surrotate clip
			surrogate_clip = adv_norm * likelihood_ratio_clip
			surrogate = torch.min(surrogate_default, surrogate_clip)
			objective = surrogate + self._alpha * e_pred
			# optimization loop
			self._actor.zero_grad()
			pi_loss = - objective.mean()
			pi_loss.backward() # retain_graph=True
			self._actor_optim.step()
		# update old policy!
		self._hard_update(self._actor, self._old_actor)

	@torch.no_grad()
	def get_action(self, x, exploit=False, warmup=False):
		if warmup:
			u = self._env.action_space.sample()
		else:
			x_torch = torch.from_numpy(x).unsqueeze(0).float().to(self._device)
			u_torch, _ = self._actor(x_torch)
			u = u_torch.cpu().numpy().squeeze(0)
		return u

	def train_mode(self):
		self._actor.train()
		self._critic.train()
		self._old_actor.eval()

	def eval_mode(self):
		self._actor.eval()
		self._critic.eval()
		self._old_actor.eval()

	def save_ckpt(self):
		torch.save(self._actor.state_dict(), os.path.join(self._save_path, "actor.pth"))
		torch.save(self._critic.state_dict(), os.path.join(self._save_path, "critic.pth"))

	def load_ckpt(self):
		try:
			self._actor.load_state_dict(torch.load(os.path.join(self._save_path, "actor.pth"), map_location=self.device))
		except:
			print("Actor checkpoint cannot be loaded.")
		try:
			self._critic.load_state_dict(torch.load(os.path.join(self._save_path, "critic.pth"), map_location=self.device))
		except:
			print("Critic checkpoints cannot be loaded.")              

	def _hard_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(local_param.data)