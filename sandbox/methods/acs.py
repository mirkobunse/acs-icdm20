import networkx as nx # used for PageRankLabelScore
import numpy as np
from sandbox import data, methods # import from this project
import sklearn.model_selection

#
# Helpers for the outside world
#
def select_from_pool(y, scores, N=1, log_msg=None):
	"""Select N instances from the pool of labels y, according to the label scores"""
	if N < 1:
		raise ValueError('N must be >= 1')
	elif len(y) < N:
		raise ValueError('Can not select N={} samples from a pool of size {}'.format(
			N, len(y)
		))
	elif np.any(scores[np.logical_not(np.isneginf(scores))] < 0):
		raise ValueError('Can not select samples with negative scores {}'.format(
			scores
		)) # still allow -Inf to indicate labels that should never be sampled

	# derive suitable sampling proportions from label scores
	N_c = data.label_counts(y, C=len(scores))
	p = np.zeros_like(scores) # sampling proportions
	if np.any(np.isposinf(scores)):
		p[np.isposinf(scores)] = 1 # sample uniformly over +Inf scores
	elif np.any(np.isneginf(scores)) and np.all(scores[np.isfinite(scores)] == 0):
		p[np.isfinite(scores)] = 1 # scores contain only -Infs, NaNs and zeros
	elif not np.all(np.isfinite(scores)):
		p[np.isfinite(scores)] = scores[np.isfinite(scores)] # NaNs and -Infs remain 0
	else:
		p = scores.copy() # wow, everything's fine!
	if np.all(p[N_c > 0] == 0):
		p = np.ones_like(p) # sample whatever remains
		p[np.logical_or(N_c == 0, np.isneginf(scores))] = 0

	return data.sample_proportions(
		y, p, max_n_samples=N, min_n_samples=N, shuffle_before=False
	) # sample from the pool using the corrected scores

#
# This supertype of all ACS strategies defines the ACS interface.
#
class LabelScore:
	"""Scoring of labels used in active class selection.

	Args:
	  X: Feature matrix of the initial training set
	  y: Labels of the initial training set (integers from 0 to C, the number of classes)
	  **kwargs: Any optional arguments which parametrize the scoring strategy

	Conventional Kwargs:
	  C: The number of classes (by default, infer them from the initial training data)
	  warmup: The number of samples per class to acquire uniformly in the beginning (default: 1)
	  verbose: The verbosity of log messages (default: 0)
	  logging_name: The identifier to use when printing log messages to stdout

	Methods:
	  add_data(X, y): Add labeled data to the initial training set.
	  score_labels(n_to_add=1): Return a C-dimensional vector of label scores, depending
	                            on the number of examples to be acquired."""
	def __init__(self, X, y, **kwargs):
		raise NotImplementedError()
	def add_data(self, X, y):
		raise NotImplementedError() # add data to the initial training set
	def score_labels(self, n_to_add=1):
		raise NotImplementedError()

#
# Heuristic strategies by Lomasky et al
#
class ProportionalLabelScore(LabelScore):
	"""Sample data in proportion to some prior assumption [lomasky2007active].

	Optional Kwargs:
	  proportions: Array of assumed class proportions. If None, assume
	               uniform proportions.
	  force_proportions: Whether the assumed proportions are enforced
	                     based on existing data (default: True).
	  crisp_decision: Whether only the most urgent class is assigned a
	                  non-zero score (default: False).

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, C=None, proportions=None, force_proportions=True, crisp_decision=False, **kwargs):
		if C is None:
			C = np.max(y) + 1 # the number of classes
		self.C = C
		if proportions is None:
			self.proportions = np.ones(C) / C # uniform
		else:
			self.proportions = proportions / proportions.sum() # normalize
		self.force_proportions = force_proportions
		if force_proportions:
			self.y = y # no need to store y if force_proportions is False
		self.crisp_decision = crisp_decision
	def add_data(self, X, y):
		if self.force_proportions:
			self.y = np.concatenate((self.y, y))
	def score_labels(self, n_to_add=1):
		if n_to_add < 1:
			raise ValueError('n_to_add must be >= 1')
		if self.force_proportions:
			# find the number of samples to acquire from each class
			N_c = data.label_counts(self.y, self.C) # n_samples per class
			missing = self.proportions * (n_to_add + len(self.y)) - N_c
			if not self.crisp_decision:
				acquisition_plan = np.maximum(missing, 0)
				acquisition_plan /= acquisition_plan.sum() # probabilities of acquisition
			else:
				acquisition_plan = np.zeros(self.C)
				acquisition_plan[np.argmax(missing)] = 1 # choose only the most urgent class
			return acquisition_plan
		else:
			return self.proportions # ignore the existing training data

class UniformLabelScore(LabelScore):
	"""Sample uniformly over all classes [lomasky2007active].

	Optional Kwargs:
	  force_equality: Select the minority class (default) or choose randomly

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, C=None, force_equality=True, **kwargs):
		if C is None:
			C = np.max(y) + 1 # the number of classes
		self.C = C
		self.force_equality = force_equality
		if force_equality:
			self.y = y
	def add_data(self, X, y):
		if self.force_equality:
			self.y = np.concatenate((self.y, y))
	def score_labels(self, n_to_add=1):
		if n_to_add < 1:
			raise ValueError('n_to_add must be >= 1')
		if self.force_equality:
			label_counts = data.label_counts(self.y, self.C)
			missing = label_counts.max() - label_counts # missing samples per class
			if missing.sum() == 0:
				return np.ones(self.C) / self.C # uniform scores
			elif missing.sum() < n_to_add:
				missing = missing + np.ones(self.C) * (n_to_add - missing.sum()) / self.C
			return missing / missing.sum()
		else:
			return np.ones(self.C) / self.C # uniform scores

class InverseLabelScore(LabelScore):
	"""Score labels by their inverse CV accuracy of the last iteration
	[lomasky2007active].

	Optional Kwargs:
	  classifier: The classifier of which the accuracy is evaluated (by default,
	              use a ParzenWindowClassifier with bandwidth 0.05)
	  K: The number of cross validation folds in each step

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, C=None, warmup=1, classifier=None, preprocessing=None, K=10, **kwargs):
		self.X = X
		self.y = y
		if C is None:
			C = np.max(y) + 1 # the number of classes
		self.C = C
		self.warmup = warmup
		if classifier is None:
			classifier = methods.ParzenWindowClassifier(C=C, bandwidth=0.05)
		self.classifier = classifier
		self.preprocessing = preprocessing
		self.K = K
	def add_data(self, X, y):
		self.X = np.concatenate((self.X, X))
		self.y = np.concatenate((self.y, y))
	def score_labels(self, n_to_add=1):
		n_missing = np.maximum(self.warmup - data.label_counts(self.y, self.C), 0)
		if n_missing.sum() > 0:
			return n_missing # warm-up phase: acquire the required number of samples per class
		acc = methods.classwise_cv(
			self.classifier,
			self.preprocessing.fit(self.X, self.y).transform(self.X) if self.preprocessing is not None else self.X,
			self.y,
			K = self.K
		)
		scr = np.ones(len(acc)) * np.inf # inf for missing classes
		scr[acc > 0] = 1 / acc[acc > 0]
		scr[acc > 0] = scr[acc > 0] / scr[acc > 0].sum() # normalize
		return scr # return the scores

class ImprovementLabelScore(LabelScore):
	"""Score labels by their latest improvement in CV accuracy [lomasky2007active].
	The first round performs an inverse scoring (see InverseLabelScore).

	Optional Kwargs:
	  classifier: The classifier of which the accuracy is evaluated (by default,
	              use a ParzenWindowClassifier with bandwidth 0.05)
	  K: The number of cross validation folds in each step

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, C=None, warmup=1, classifier=None, preprocessing=None, K=10, **kwargs):
		self.X = X
		self.y = y
		if C is None:
			C = np.max(y) + 1 # the number of classes
		self.C = C
		self.warmup = warmup
		if classifier is None:
			classifier = methods.ParzenWindowClassifier(C=C, bandwidth=0.05)
		self.classifier = classifier
		self.preprocessing = preprocessing
		self.K = K
		self.prevAcc = None # accuracy of the previous round
	def add_data(self, X, y):
		self.X = np.concatenate((self.X, X))
		self.y = np.concatenate((self.y, y))
	def score_labels(self, n_to_add=1):
		acc = methods.classwise_cv(
			self.classifier,
			self.preprocessing.fit(self.X, self.y).transform(self.X) if self.preprocessing is not None else self.X,
			self.y,
			K = self.K
		)
		scr = np.ones(self.C) * np.inf # initial scores with inf for missing classes
		if self.prevAcc is not None:
			scr[acc > 0] = acc[acc > 0] - self.prevAcc[acc > 0] # absolute improvement
			if scr[acc > 0].sum() > 0:
				scr[acc > 0] = scr[acc > 0] / scr[acc > 0].sum() # normalize
				scr[acc > 0] = np.maximum(0, scr[acc > 0]) # the improvement must be >= 0
			elif np.any(acc > 0):
				scr[acc > 0] = np.ones(np.sum(acc > 0)) / np.sum(acc > 0)
			else:
				scr = np.ones(self.C) / self.C # uniform sampling
		else: # InverseLabelScore.score_labels in the first round
			scr[acc > 0] = 1 / acc[acc > 0]
			scr[acc > 0] = scr[acc > 0] / scr[acc > 0].sum() # normalize
		self.prevAcc = acc # update accuracies
		n_missing = np.maximum(self.warmup - data.label_counts(self.y, self.C), 0)
		if n_missing.sum() > 0:
			return n_missing # warm-up phase: acquire the required number of samples per class
		else:
			return scr # return the scores

class RedistrictionLabelScore(LabelScore):
	"""Score labels by the redistriction strategy [lomasky2007active].
	The first round performs an inverse scoring (see InverseLabelScore).

	Optional Kwargs:
	  classifier: The classifier of which the accuracy is evaluated (by default,
	              use a ParzenWindowClassifier with bandwidth 0.05)
	  K: The number of cross validation folds in each step

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, C=None, warmup=1, classifier=None, preprocessing=None, K=10, **kwargs):
		self.X = X
		self.y = y
		if C is None:
			C = np.max(y) + 1 # the number of classes
		self.C = C
		self.warmup = warmup
		if classifier is None:
			classifier = methods.ParzenWindowClassifier(C=C, bandwidth=0.05)
		self.classifier = classifier
		self.preprocessing = preprocessing
		self.K = K
		self.prevPred = None # accuracy of the previous round
	def add_data(self, X, y):
		self.X = np.concatenate((self.X, X))
		self.y = np.concatenate((self.y, y))
	def score_labels(self, n_to_add=1):
		_X = self.preprocessing.fit(self.X, self.y).transform(self.X) if self.preprocessing is not None else self.X
		cross_val = sklearn.model_selection.KFold(
			min(self.K, len(_X)), shuffle=False
		) # cross-validation split with at most n_samples folds
		pred = sklearn.model_selection.cross_val_predict(
			self.classifier, _X, self.y, cv=cross_val
		) # out-of-bag predictions for all samples
		scr = np.ones(self.C) * np.inf # np.inf for missing classes
		if self.prevPred is not None:
			y = self.y[:len(self.prevPred)] # target values which are relevant now
			pred_changed = pred[:len(self.prevPred)] != self.prevPred # prediction changed
			for label in np.unique(y):
				scr[label] = np.sum(pred_changed[y == label]) # redistrictions per label
		else: # InverseLabelScore.score_labels in the first round
			acc = methods.classwise_cv(self.classifier, _X, self.y, K=self.K)
			scr[acc > 0] = 1 / acc[acc > 0]
			scr[acc > 0] = scr[acc > 0] / scr[acc > 0].sum() # normalize
		self.prevPred = pred # update predictions
		n_missing = np.maximum(self.warmup - data.label_counts(self.y, self.C), 0)
		if n_missing.sum() > 0:
			return n_missing # warm-up phase: acquire the required number of samples per class
		else:
			return scr # return the scores

#
# Other strategies
#
class IgnorantLabelScore(LabelScore):
	"""Ignore one class that is known to be easy and score the remaining classes.

	Optional Kwargs:
	  ignore_class: The class that is known to be easy
	  actual_score: The scoring used for the remaining classes, parametrized by
	                the remaining **kwargs

	For more information, see sandbox.methods.acs.LabelScore."""
	def __init__(self, X, y, ignore_class=0, actual_score=None, **kwargs):
		self.ignore_class = ignore_class
		X, y = self.__ignore_class(X, y)
		if actual_score is None:
			actual_score = UniformLabelScore(X, y, **kwargs)
		else:
			actual_score.X = X # update data of existing actual_score
			actual_score.y = y
		actual_score.C -= 1 # there is one class less to consider
		self.actual_score = actual_score
		if 'warmup' in kwargs:
			print('WARNING: OptimalLabelScore does intendedly not support warmup')
	def __ignore_class(self, X, y):
		X = X[y != self.ignore_class]
		y = y[y != self.ignore_class]
		y[y >= self.ignore_class] -= 1
		return X, y
	def add_data(self, X, y):
		self.actual_score.add_data(*(self.__ignore_class(X, y)))
	def score_labels(self, n_to_add=1):
		return np.insert(
			self.actual_score.score_labels(n_to_add),
			self.ignore_class,
			-np.inf
		)
