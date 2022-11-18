
import time
import os
from copy import deepcopy
import torch
from torch import nn

from fastrl.agent import Agent


class SoftActorCritic(Agent):
	
	_agent_name = "sac"

	def __init__(self, env, actor_nn, critic_nn, optimizer_args, 
			gamma=0.99, alpha=0.05, tau=0.005, batch_size=256, 
			**kwargs):
		super(SoftActorCritic, self).__init__(env, **kwargs)
		self._env = env
		self._alpha = alpha
		self._gamma = gamma
		self._tau = tau
		self._batch_size = batch_size
		self._critic_loss_fcn = nn.MSELoss()
		#
		self._actor = actor_nn(self._nx, self._nu).to(self._device)
		self._critic_1 = critic_nn(self._nx, self._nu).to(self._device)
		self._critic_2 = critic_nn(self._nx, self._nu).to(self._device)
		self._critic_1_target = critic_nn(self._nx, self._nu).to(self._device)
		self._critic_2_target = critic_nn(self._nx, self._nu).to(self._device)
		self._hard_update(self._critic_1, self._critic_1_target) # hard update at the beginning
		self._hard_update(self._critic_2, self._critic_2_target) # hard update at the beginning
		#
		self._actor_optim = torch.optim.Adam(self._actor.parameters(), **optimizer_args)
		self._critic_1_optim = torch.optim.Adam(self._critic_1.parameters(), **optimizer_args)
		self._critic_2_optim = torch.optim.Adam(self._critic_2.parameters(), **optimizer_args)

	def learn_on_step(self):
		# pass
		if self._batch_size > self._experience_memory.size:
			return None
		x, u, r, xp, done = self._experience_memory.sample_random(self._batch_size)
		# generate q targets
		with torch.no_grad():
			# print(xp)
			up_pred, up_pred_dist = self._actor(xp)
			ep_pred = - up_pred_dist.log_prob(up_pred).sum(dim=-1, keepdim=True)
			qp_1_target = self._critic_1_target(xp, up_pred)
			qp_2_target = self._critic_2_target(xp, up_pred)
			qp_target = torch.min(qp_1_target, qp_2_target) + self._alpha * ep_pred
			# q_target = r.unsqueeze(-1) + (self._gamma * qp_target * (1.0-done))
			q_target = r.unsqueeze(-1) + (self._gamma * qp_target)
		# update critic 1
		self._critic_1_optim.zero_grad()
		q_1 = self._critic_1(x, u)
		q_1_loss = 0.5*self._critic_loss_fcn(q_1, q_target)
		q_1_loss.backward()
		self._critic_1_optim.step()
		# update critic 2
		self._critic_2_optim.zero_grad()
		q_2 = self._critic_2(x, u)   
		q_2_loss = 0.5*self._critic_loss_fcn(q_2, q_target)
		q_2_loss.backward()
		self._critic_2_optim.step()
		#update actor
		self._actor_optim.zero_grad()
		u_pred, u_pred_dist = self._actor(x)
		e_pred = - u_pred_dist.log_prob(u_pred).sum(dim=-1, keepdim=True)
		# print(e_pred)
		q_pi = torch.min(self._critic_1(x, u_pred), self._critic_2(x, u_pred))
		pi_loss = -(q_pi + self._alpha * e_pred).mean()
		pi_loss.backward()
		self._actor_optim.step()
		# soft update of target critic networks
		self._soft_update(self._critic_1, self._critic_1_target)
		self._soft_update(self._critic_2, self._critic_2_target)

	def learn_on_episode(self):
		pass
		#if self._batch_size > self._experience_memory.size:
		#	return None
		#for i in range(self._t): 
		#	x, u, r, xp, done = self._experience_memory.sample_random(self._batch_size)
		#	# generate q targets
		#	with torch.no_grad():
		#		# print(xp)
		#		up_pred, up_pred_dist = self._actor(xp)
		#		ep_pred = - up_pred_dist.log_prob(up_pred).sum(dim=-1, keepdim=True)
		#		qp_1_target = self._critic_1_target(xp, up_pred)
		#		qp_2_target = self._critic_2_target(xp, up_pred)
		#		qp_target = torch.min(qp_1_target, qp_2_target) + self._alpha * ep_pred
		#		# q_target = r.unsqueeze(-1) + (self._gamma * qp_target * (1.0-done))
		#		q_target = r.unsqueeze(-1) + (self._gamma * qp_target)
		#	# update critic 1
		#	self._critic_1_optim.zero_grad()
		#	q_1 = self._critic_1(x, u)
		#	q_1_loss = 0.5*self._critic_loss_fcn(q_1, q_target)
		#	q_1_loss.backward()
		#	self._critic_1_optim.step()
		#	# update critic 2
		#	self._critic_2_optim.zero_grad()
		#	q_2 = self._critic_2(x, u)   
		#	q_2_loss = 0.5*self._critic_loss_fcn(q_2, q_target)
		#	q_2_loss.backward()
		#	self._critic_2_optim.step()
		#	#update actor
		#	self._actor_optim.zero_grad()
		#	u_pred, u_pred_dist = self._actor(x)
		#	e_pred = - u_pred_dist.log_prob(u_pred).sum(dim=-1, keepdim=True)
		#	# print(e_pred)
		#	q_pi = torch.min(self._critic_1(x, u_pred), self._critic_2(x, u_pred))
		#	pi_loss = -(q_pi + self._alpha * e_pred).mean()
		#	pi_loss.backward()
		#	self._actor_optim.step()
		#	# soft update of target critic networks
		#	self._soft_update(self._critic_1, self._critic_1_target)
		#	self._soft_update(self._critic_2, self._critic_2_target)

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
		self._critic_1.train()
		self._critic_2.train()

	def eval_mode(self):
		self._actor.eval()
		self._critic_1.eval()
		self._critic_2.eval()

	def save_ckpt(self):
		torch.save(self._actor.state_dict(), os.path.join(self._save_path, "actor.pth"))
		torch.save(self._critic_1.state_dict(), os.path.join(self._save_path, "critic_1.pth"))
		torch.save(self._critic_2.state_dict(), os.path.join(self._save_path, "critic_2.pth"))

	def load_ckpt(self):
		try:
			self._actor.load_state_dict(torch.load(os.path.join(self._save_path, "actor.pth"), map_location=self.device))
		except:
			print("Actor checkpoint cannot be loaded.")
		try:
			self._critic_1.load_state_dict(torch.load(os.path.join(self._save_path, "critic_1.pth"), map_location=self.device))
			self._critic_2.load_state_dict(torch.load(os.path.join(self._save_path, "critic_2.pth"), map_location=self.device))
			self._hard_update(self._critic_1, self._critic_1_target) # hard update after loading
			self._hard_update(self._critic_2, self._critic_2_target) # hard update after loading
		except:
			print("Critic checkpoints cannot be loaded.")              

	def _soft_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)

	def _hard_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(local_param.data)
