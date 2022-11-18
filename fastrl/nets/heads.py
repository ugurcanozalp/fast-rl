
import time
import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical, TransformedDistribution 
from torch.distributions.transforms import TanhTransform, AffineTransform, ComposeTransform

class GaussianHead(nn.Module):
	def __init__(self, n):
		super(GaussianHead, self).__init__()
		self._n = n

	def forward(self, x):
		mean = x[...,:self._n]
		logvar = x[...,self._n:]
		std = (0.5*logvar).exp() 
		# print(std)
		dist = Normal(mean, std, validate_args=False)
		y = dist.rsample()
		return y, dist

class SquashedGaussianHead(nn.Module):
	def __init__(self, n, scale=1.0):
		super(SquashedGaussianHead, self).__init__()
		self._n = n
		self._scale = scale

	def forward(self, x):
		# bt means before tanh
		mean_bt = x[...,:self._n]
		logvar_bt = x[...,self._n:]
		std_bt = (0.5*logvar_bt).exp() 
		dist_bt = Normal(mean_bt, std_bt, validate_args=False)
		transform = ComposeTransform([
			AffineTransform(0., 1.0/self._scale, cache_size=1), 
			TanhTransform(cache_size=1), 
			AffineTransform(0., self._scale, cache_size=1)],
			cache_size=1) 
		dist = TransformedDistribution(dist_bt, [transform])
		y = dist.rsample()
		return y, dist

class CategoricalHead(nn.Module):
	def __init__(self, n):
		super(CategoricalHead, self).__init__()
		self._n = n

	def forward(self, x):
		logit = x
		probs = nn.functional.softmax(logit)
		dist = Categorical(probs, validate_args=False)
		y = dist.rsample()
		return y, dist
		
class DeterministicHead(nn.Module):
	def __init__(self, n):
		super(DeterministicHead, self).__init__()
		self._n = n

	def forward(self, x):
		mean = x
		y = mean
		dist = None
		return y, dist
