
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
	with open(path) as fp:
		data = json.load(fp)
	return data

def plot_history(history, fname=None):
	"""Plot training results.
	
	Args:
	    history (Dict): Training history object
	"""
	episode_time_steps = np.array(history["episode_time_steps"])
	episode_scores = np.array(history["episode_scores"])
	episode_ma_scores = np.array(history["episode_ma_scores"])
	episode_ma_std_scores = np.array(history["episode_ma_std_scores"])
	test_time_steps = np.array(history["test_time_steps"])
	test_average_scores = np.array(history["test_average_scores"])
	# 
	auc_score = episode_scores.mean()
	max_score = episode_scores.max()
	last_score = episode_scores[-1]
	print(f"AUC of Episodic Score: {auc_score}")
	print(f"Max of Episodic Score: {max_score}")
	print(f"Last Episodic Score: {last_score}")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(episode_time_steps, episode_scores,'r.',alpha=0.3, label="score")
	ax.plot(episode_time_steps, episode_ma_scores,'b-',alpha=1.0, label="ma_score")
	ax.fill_between(episode_time_steps, 
		episode_ma_scores - episode_ma_std_scores,
		episode_ma_scores + episode_ma_std_scores,
		facecolor='b', alpha=0.5)
	ax.scatter(test_time_steps, test_average_scores, label="test_avg")
	ax.set_ylabel("Score")
	ax.set_xlabel("Time Step #")
	ax.set_title("Score History")
	ax.legend()
	if fname is not None:
		fig.savefig(fname+".png")
	fig.show()


def plot_multiple_history(histories, fname=None):
	"""Plot multiple training results.
	
	Args:
	    history (List[Dict]): Training history object
	"""
	episode_time_steps = np.array([history["episode_time_steps"] for history in histories])
	episode_scores = np.array([history["episode_scores"] for history in histories])
	episode_ma_scores = np.array([history["episode_ma_scores"] for history in histories])
	# 
	time_steps = episode_time_steps.mean(0)
	mean_score = episode_ma_scores.mean(0)
	std_score = episode_ma_scores.std(0)
	# 
	auc_score = mean_score.mean()
	max_score = mean_score.max()
	last_score = mean_score[-1]
	print(f"AUC of Episodic Score: {auc_score}")
	print(f"Max of Episodic Score: {max_score}")
	print(f"Last Episodic Score: {last_score}")
	# 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(time_steps, mean_score,'b-',alpha=1.0, label="ma_score")
	ax.fill_between(time_steps, 
		mean_score - std_score,
		mean_score + std_score,
		facecolor='b', alpha=0.5)
	ax.set_ylabel("Score")
	ax.set_xlabel("Time Step #")
	ax.set_title("Score History")
	ax.legend()
	if fname is not None:
		fig.savefig(fname+".png")
	fig.show()

def plot_multiple_histories_list(histories_list, exp_names, fname=None):
	"""Plot multiple training results.
	
	Args:
	    history (List[Dict]): Training history object
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for histories, exp_name in zip(histories_list, exp_names):
		episode_time_steps = np.array([history["episode_time_steps"] for history in histories])
		episode_scores = np.array([history["episode_scores"] for history in histories])
		episode_ma_scores = np.array([history["episode_ma_scores"] for history in histories])
		# 
		time_steps = episode_time_steps.mean(0)
		mean_score = episode_ma_scores.mean(0)
		std_score = episode_ma_scores.std(0)
		# 
		auc_score = mean_score.mean()
		max_score = mean_score.max()
		last_score = mean_score[-1]
		print(f"AUC of Episodic Score: {auc_score}")
		print(f"Max of Episodic Score: {max_score}")
		print(f"Last Episodic Score: {last_score}")
		ax.plot(time_steps, mean_score,'-',alpha=1.0, label=exp_name)
		#ax.fill_between(time_steps, 
		#	mean_score - std_score,
		#	mean_score + std_score,
		#	facecolor='b', alpha=0.5)
		ax.set_ylabel("Score")
		ax.set_xlabel("Time Step #")
		ax.set_title("Score History")
		ax.legend()
	if fname is not None:
		fig.savefig(fname+".png")
	fig.show()


def analyze_results(path):
	if isinstance(path, str):
		history = load_json(path)
		plot_history(history, "")
	elif isinstance(path, list):
		histories = [load_json(p) for p in path]
		plot_multiple_history(histories, "")