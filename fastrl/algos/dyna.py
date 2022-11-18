
import time
import os
import math
import random
import torch
from torch.distributions import Normal
from torch import nn

from fastrl.agent import Agent


class Dyna(Agent):
	
	_agent_name = "dyna"

	def __init__(self, env, model_nn, policy_nn, optimizer_args, 
			horizon = 20, batch_size = 256, alpha=0.05, 
			**kwargs):
		super(Dyna, self).__init__(env, **kwargs)
		self._alpha = alpha
		self._horizon = horizon # policy horizon
		self._batch_size = batch_size
		self._model = model_nn(self._nx, self._nu).to(self._device) # Model network
		self._policy = policy_nn(self._nx, self._nu).to(self._device) # Policy network
		self._model_optim = torch.optim.Adam(self._model.parameters(), **optimizer_args) # optimizer of model paraameters
		self._policy_optim = torch.optim.Adam(self._policy.parameters(), **optimizer_args) # optimizer of policy paraameters

	def learn_on_step(self):
		if self._batch_size > self._experience_memory.size:
			return None
		self._train_model()
		self._train_policy()	

	def learn_on_episode(self):
		pass

	def _train_model(self):
		x, u, r, xp, done = self._experience_memory.sample_random(self._batch_size)
		dx_r = torch.concat([xp - x, r.unsqueeze(-1)], dim=-1)
		self._model.zero_grad()
		dx_r_pred, dx_r_pred_dist = self._model(x, u)
		loss = - dx_r_pred_dist.log_prob(dx_r).mean() # negative log probability is the loss, mean over batches, states etc. 
		# print(f"loss: {loss}")
		loss.backward()
		self._model_optim.step()	

	def _train_policy(self):
		x, _, _, _, _ = self._experience_memory.sample_random(self._batch_size)
		policy_loss = 0 # Implement
		for t in range(self._horizon):
			u_pred, u_pred_dist = self._policy(x) # batch_size, num_particles, nu
			dx_r_pred, _ = self._model(x, u_pred)
			xp_pred = dx_r_pred[...,:-1] + x
			r_pred = dx_r_pred[...,-1]
			policy_loss -= r_pred.mean()
			policy_loss += self._alpha * u_pred_dist.log_prob(u_pred).sum(dim=-1, keepdim=True).mean() # entropy maximization term
			# Continue simulation on learned model (gradients are passed!)
			x = xp_pred
		policy_loss /= self._horizon
		self._policy_optim.zero_grad()
		policy_loss.backward()
		self._policy_optim.step()

	@torch.no_grad()
	def get_action(self, x, exploit = False, warmup = False):
		if warmup:
			u = self._env.action_space.sample()
		else:
			x_torch = torch.from_numpy(x).float().to(self._device)
			u_torch, _ = self._policy(x_torch)
			u = u_torch.cpu().numpy()
		return u

	def train_mode(self):
		self._model.train()
		self._policy.train()

	def eval_mode(self):
		self._model.eval()
		self._policy.eval()

	def save_ckpt(self):
		torch.save(self._model.state_dict(), os.path.join(self._save_path, "dynamics.pth"))
		torch.save(self._policy.state_dict(), os.path.join(self._save_path, "policy.pth"))

	def load_ckpt(self):
		try:
			self._model.load_state_dict(torch.load(os.path.join(self._save_path, "dynamics.pth"), map_location=self.device))
			self._policy.load_state_dict(torch.load(os.path.join(self._save_path, "policy.pth"), map_location=self.device))
		except:
			print("Dynamics and policy model checkpoint cannot be loaded.")      

	def episode_start(self):
		pass

	def _rollout(self):
		pass
