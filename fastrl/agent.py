"""Generic RL Agent definition
"""
import os
import time
import math
import numpy as np
import json
from copy import deepcopy
from datetime import datetime
import torch
import matplotlib.pyplot as plt 
import gym 
from gym.wrappers import RecordVideo
from .experience_memory import ExperienceMemory

class Agent:

	"""Abstract class for all RL agents.
	"""
	
	_agent_name = "abstract"

	def __init__(self, env: gym.Env, device: str = "cpu", **kwargs):
		"""Summary
		
		Args:
		    env (gym.Env): Gym environment to work on
		    device (str, optional): Device in which algorithm works on (cpu, cuda, etc.)
		    **kwargs: Other arguments such as buffer size etc.
		"""
		self._device = device
		self._env = env
		self._env_monitor = None
		if hasattr(self._env, "name"):
			self._env_name = self._env.name 
		else:
			self._env_name = self._env.spec.id
		self._save_path = os.path.join(".", "results", self._agent_name, self._env_name)
		if not os.path.exists(self._save_path):
			os.makedirs(self._save_path)
		self._nx, self._nu = self._env.observation_space.shape, self._env.action_space.shape
		self._nx_flat, self._nu_flat = np.prod(self._nx), np.prod(self._nu)
		self._u_min = torch.from_numpy(self._env.action_space.low).float().to(self._device)
		self._u_max = torch.from_numpy(self._env.action_space.high).float().to(self._device)
		self._x_min = torch.from_numpy(self._env.observation_space.low).float().to(self._device)
		self._x_max = torch.from_numpy(self._env.observation_space.high).float().to(self._device)
		self._experience_memory = ExperienceMemory(device=self._device, **kwargs)
		self._T = self._env._max_episode_steps
		self._t = 0
		# Training related data
		self._t_total = 0
		self._clear_record()
	
	def _clear_record(self, x = None):
		"""The agent saves state-action history, this function clears the records.
		
		Args:
		    x (None, optional): Initial state after reset call.
		"""
		self._x = np.zeros((self._T+1, *self._nx), dtype=np.float32)
		if x is not None:
			self._x[0] = x # clear while adding initial state to record
		self._u = np.zeros((self._T, *self._nu), dtype=np.float32)
		self._r = np.zeros(self._T, dtype=np.float32)
		self._d = np.ones(self._T, dtype=np.bool_)

	def _record(self, xp, u, r, d, t):
		"""This function is used to record transition tuple onto a time point.
		
		Args:
		    xp (TYPE): Next state
		    u (TYPE): Action
		    r (TYPE): Reward
		    d (TYPE): Done flag
		    t (TYPE): The time index to save transition
		"""
		self._x[t+1] = xp
		self._u[t] = u
		self._r[t] = r
		self._d[t] = d

	def train_mode(self):
		"""Override this function if you need to do something before training.
		"""
		pass

	def test_mode(self):
		"""Override this function if you need to do something before training.
		"""
		pass

	def save_ckpt(self):
		"""Override this function if you need to do save something about your agent.
		"""
		pass

	def load_ckpt(self):
		"""Override this function if you need to do load saved model.
		"""
		pass

	def learn_on_step(self):
		"""Define what to do after each time step for learning.
		
		Raises:
		    NotImplementedError: This function must be overriden. 
		"""
		raise NotImplementedError

	def learn_on_episode(self):
		"""Define what to do after each episode for learning.
		
		Raises:
		    NotImplementedError: This function must be overriden. 
		"""
		raise NotImplementedError

	def get_action(self, warmup=False, exploit=False):
		"""Take action according to your current model.
		
		Args:
		    warmup (bool, optional): Flag for warmup steps
		    exploit (bool, optional): Exploit flag.
		
		Raises:
		    NotImplementedError: This function must be overriden. 
		"""
		raise NotImplementedError

	def episode_end(self):
		"""Override this function if you need to do something after episode end.
		"""
		pass

	def step_end(self):
		"""Override this function if you need to do something after step end.
		"""
		pass

	def episode_start(self):
		"""Override this function if you need to do something after episode start.
		"""
		pass

	def step_start(self):
		"""Override this function if you need to do something after step start.
		"""
		pass

	def train(self, max_num_episodes = 500, warmup_episodes = 1, test_interval = None, 
			num_episodes_per_test = 10, test_score_limit = None, plot = True):
		"""Main train loop.
		
		Args:
		    max_num_episodes (int, optional): Maximum number of episodes for training.
		    warmup_episodes (int, optional): Number of episodes for warm up.
		    test_interval (None, optional): Defines model test frequency 
		    num_episodes_per_test (int, optional): Number of episodes for a test
		    test_score_limit (None, optional): Limit for testing score. If this limit exceed, training stops.
		    render (bool, optional): Render flag
		    plot (bool, optional): Plot flag after training.
		
		Returns:
		    TYPE: Description
		"""
		episode_scores = []
		episode_ma_scores = []
		episode_ma_std_scores = []
		episode_time_steps = []
		test_average_scores = []
		test_time_steps = []
		episode = 0 # Reset episode counter before episode starts
		self._t_total = 0
		while episode < max_num_episodes:
			warmup = episode < warmup_episodes
			self.train_mode()
			x, _ = self._env.reset()
			truncated = False
			self._t = 0 # Reset time counter before episode starts
			score = 0
			self._clear_record(x)
			self.episode_start()
			while not truncated: 
				self.step_start()
				u = self.get_action(x, exploit = False, warmup = warmup)
				xp, r, d, truncated, info = self._env.step(u)
				self._record(xp, u, r, d, self._t)
				self._experience_memory.add(x, u, r, xp, d)
				x = xp.copy()
				self._t += 1
				self._t_total += 1
				score += r
				self.step_end()
				self.learn_on_step()
			self.episode_end()
			if not warmup:
				self.learn_on_episode()
			episode += 1
			episode_scores.append(float(score))
			episode_time_steps.append(self._t_total)
			average_span = min(episode, max_num_episodes//50 + 1)
			episode_scores_span = [episode_scores[i] for i in range(-1, -average_span-1, -1)] 
			average_score = sum(episode_scores_span) / average_span
			episode_ma_scores.append(average_score)
			episode_scores_sq_span = [(episode_scores[i]-episode_ma_scores[i])**2 for i in range(-1, -average_span-1, -1)] 
			average_std_score = math.sqrt( sum(episode_scores_sq_span) / average_span )
			episode_ma_std_scores.append(average_std_score)
			print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, average_score, score), end="")
			if (test_interval is not None) and (episode % test_interval == 0):
				self.test_mode() # test in eval mode.
				_, test_average_score = self.test(num_episodes=num_episodes_per_test)
				test_average_scores.append(test_average_score)
				ckpt_name = "".join([self._agent_name, self._env_name, 'Episode: '+str(episode)])
				self.save_ckpt()
				test_time_steps.append(self._t_total)
				if (test_score_limit is not None) and test_average_score > test_score_limit:
					break
		history = self._history(episode_scores, episode_ma_scores, episode_ma_std_scores, episode_time_steps, test_average_scores, test_time_steps)
		date = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
		fname = os.path.join(self._save_path, date)
		with open(fname+".json", "w") as fp:
			json.dump(history, fp, indent=4)
		if plot:
			self._plot_history(history, fname=fname)
		return history

	def test(self, num_episodes = 1, exploit = True, monitor = False):
		"""Main test loop.
		
		Args:
		    num_episodes (int, optional): Number of episodes for training.
		    exploit (bool, optional): Exploit flag.
		    render (bool, optional): Render flag.
		    monitor (bool, optional): Monitoring flag for recording an episode.
		
		Returns:
		    TYPE: Description
		"""
		if monitor and self._env_monitor is None:
			self._env_monitor = RecordVideo(self._env, self._save_path)
		env = self._env_monitor if monitor else self._env
		self.test_mode()
		episode_scores = []
		episode = 0 # Reset episode counter before episode starts
		while episode < num_episodes:
			episode += 1
			x, _ = env.reset()
			self._x[0] = x
			truncated = False
			score = 0
			self._t = 0 # Reset time counter before episode starts
			self._clear_record(x)
			self.episode_start()
			while not truncated: 
				self.step_start()
				u = self.get_action(x, exploit = exploit)
				xp, r, d, truncated, info = env.step(u)
				self._record(xp, u, r, d, self._t)
				x = xp
				self._t += 1
				score += r
				self.step_end()
			self.episode_end()
			episode_scores.append(float(score))
		average_score = sum(episode_scores) / num_episodes
		print('\r\nAverage Test Score: {:.2f}'.format(average_score))
		return episode_scores, average_score

	@staticmethod
	def _history(episode_scores, episode_ma_scores, episode_ma_std_scores, episode_time_steps, test_average_scores, test_time_steps):
		history = {}
		history["episode_scores"] = episode_scores
		history["episode_ma_scores"] = episode_ma_scores
		history["episode_ma_std_scores"] = episode_ma_std_scores
		history["episode_time_steps"] = episode_time_steps
		history["test_average_scores"] = test_average_scores
		history["test_time_steps"] = test_time_steps
		return history

	@staticmethod
	def _plot_history(history, fname=None):
		"""Plot training results.
		
		Args:
		    history (Dict): Training history object
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(history["episode_time_steps"], history["episode_scores"],'r.',alpha=0.3, label="score")
		ax.plot(history["episode_time_steps"], history["episode_ma_scores"],'b-',alpha=1.0, label="ma_score")
		ax.fill_between(history["episode_time_steps"], 
			[mu-std for mu, std in zip(history["episode_ma_scores"], history["episode_ma_std_scores"])], 
			[mu+std for mu, std in zip(history["episode_ma_scores"], history["episode_ma_std_scores"])], 
			facecolor='b', alpha=0.5)
		ax.scatter(history["test_time_steps"], history["test_average_scores"], label="test_avg")
		ax.set_ylabel("Score")
		ax.set_xlabel("Time Step #")
		ax.set_title("Score History")
		ax.legend()
		if fname is not None:
			fig.savefig(fname+".png")
		fig.show()
		