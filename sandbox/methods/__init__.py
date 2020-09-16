import numpy as np
import sklearn, scipy
from copy import deepcopy
from sandbox import data # import from this project
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KernelDensity, KNeighborsClassifier)
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#
# Classification
#
class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
	"""The Parzen window classifier can be interpreted as a Bayesian variant of the k-nearest
	neighbor classifier. It trains one KDE per class and predicts the class for which the
	KDE returns the highest probability.

	Chapelle: Active learning for Parzen window classifier. In: Int. Workshop on Artificial
	Intelligence and Statistics, 2005."""
	def __init__(self, C=None, uniform_classes=False, **kwargs):
		self.C = C
		self.uniform_classes = uniform_classes
		self.kwargs = kwargs
	def fit(self, X, y, sample_weight=None):
		if self.C is None:
			self.C = np.max(y) + 1 # the number of classes
		if sample_weight is None:
			sample_weight = np.ones(len(y))
		elif len(sample_weight) != len(y):
			raise ValueError('sample_weight must have the same length as y')
		else:
			sample_weight *= len(y) / sample_weight.sum() # sums up to n_samples
		self.labels = np.unique(y) # classes that actually appear
		self.kdes = [] # one KDE per class
		self.log_N = [] # log(number) of training samples in each class
		for label in self.labels:
			w_label = sample_weight[y == label] # weights of this class
			kde = KernelDensity(**self.kwargs)
			kde.fit(X[y == label], sample_weight=w_label)
			log_N = np.log(w_label.sum())
			self.kdes.append(kde)
			self.log_N.append(log_N)
		self.log_N = np.expand_dims(self.log_N, 0) # match proba.shape in predict_proba
		return self
	def predict_proba(self, X):
		proba = np.zeros((len(X), self.C))
		proba[:,self.labels] = np.stack(list(map(
			lambda kde: kde.score_samples(X),
			self.kdes
		)), axis=1) # log probabilities
		if not self.uniform_classes:
			proba[:,self.labels] += self.log_N # the default case (non-uniform)
		proba[:,self.labels] = np.exp(proba[:,self.labels]) # log-density to density
		proba_sum = np.sum(proba, axis=1, keepdims=True)
		proba_sum[proba_sum == 0] = 1. # next line will not impact these instances
		return proba / proba_sum # normalize to unit sum per instance
	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)
	def get_params(self, deep = False):
		params = deepcopy(self.kwargs) # from the copy package
		params.update({'C': self.C, 'uniform_classes': self.uniform_classes}) # add multiple pairs
		return params
	def set_params(self, **parameters):
		self.C = parameters.pop('C', None)
		self.uniform_classes = parameters.pop('uniform_classes', None)
		self.kwargs = parameters
		return self

def make_pipeline(clf_config=None, C=None, nca_components=2, input_dim=None):
	"""Create a pipeline from an (optional) NCA, a MinMaxScaler and an optional classifier."""
	if input_dim is not None and nca_components >= input_dim:
		print('WARNING: Skipping NCA because nca_components={} >= input_dim={}'.format(nca_components, input_dim))
		nca_components = -1 # do not perform an NCA if the input dimension will be smaller
	pipeline_steps = []
	if nca_components >= 0:
		pipeline_steps.append(StandardScaler())
		if nca_components > 0:
			pipeline_steps.append(NeighborhoodComponentsAnalysis(n_components=nca_components))
		else:
			pipeline_steps.append(NeighborhoodComponentsAnalysis()) # use full dimensionality
	pipeline_steps.append(MinMaxScaler()) # always perform MinMaxScaling
	if clf_config is not None:
		if clf_config['class_name'] not in ['SVC', 'LinearSVC']:
			try:
				clf = eval(clf_config['class_name'])(C=C, **clf_config['parameters'])
			except TypeError:
				clf = eval(clf_config['class_name'])(**clf_config['parameters']) # sklearn classifiers do not have a C parameter
			pipeline_steps.append(clf) # can be omitted if only the pre-processing is desired
		else:
			clf = eval(clf_config['class_name'])(**clf_config['parameters'])
			pipeline_steps.append(clf)
	return sklearn.pipeline.make_pipeline(*pipeline_steps)

#
# Evaluation
#
def evaluate_error(X_trn, y_trn, X_tst, y_tst, **kwargs):
	"""See __evaluate(sklearn.metrics.zero_one_loss, X_trn, y_trn, X_tst, y_tst, **kwargs)"""
	return __evaluate(sklearn.metrics.zero_one_loss, X_trn, y_trn, X_tst, y_tst, **kwargs)

def evaluate_classwise_error(X_trn, y_trn, X_tst, y_tst, **kwargs):
	"""See 1 - np.diag(__evaluate_confusion(X_trn, y_trn, X_tst, y_tst, **kwargs))"""
	return 1 - np.diag(__evaluate_confusion(X_trn, y_trn, X_tst, y_tst, **kwargs))

def __evaluate_confusion(X_trn, y_trn, X_tst, y_tst, C=None, normalize='truth', **kwargs):
	"""See __evaluate(sklearn.metrics.confusion_matrix, X_trn, y_trn, X_tst, y_tst, **kwargs)"""
	if C is None:
		C = max(np.max(y_trn), np.max(y_tst)) + 1 # the number of classes
	confusion = __evaluate(
		sklearn.metrics.confusion_matrix,
		X_trn, y_trn, X_tst, y_tst,
		C=C,
		labels=list(range(C)),
		**kwargs
	)
	if normalize == 'truth':
		confusion = confusion / confusion.sum(axis=1, keepdims=True)
	elif normalize == 'prediction':
		confusion = confusion / confusion.sum(axis=0, keepdims=True)
	elif normalize != None:
		raise ValueError('normalize={} is invalid'.format(normalize))
	return confusion

def __evaluate(metric, X_trn, y_trn, X_tst, y_tst, C=None, classifier=None, **kwargs):
	"""Evaluate the metric of a classifier trained on (X_trn, y_trn) by predicting
	(X_tst, y_tst)."""
	if classifier is None:
		classifier = ParzenWindowClassifier(C=C, bandwidth=0.05)
	elif isinstance(classifier, str):
		classifier = eval(classifier)
	classifier.fit(X_trn, y_trn)
	return metric(y_tst, classifier.predict(X_tst), **kwargs)

def classwise_cv(classifier, X, y, K=10, shuffle=True, metric=None):
	"""Take out a K-fold cross validation where the metric is computed for each separate class.

	If no metric is given, sklearn.metrics.accuracy_score is used."""
	if metric is None:
		metric = sklearn.metrics.accuracy_score
	C = len(np.unique(y)) # the number of classes
	K = min(K, len(X)) # cannot have more splits than samples
	scores = np.zeros((K, C))
	for k, tup in enumerate(sklearn.model_selection.KFold(K, shuffle=shuffle).split(X)):
		i_trn, i_tst = tup # unpack KFold indices
		classifier.fit(X[i_trn], y[i_trn])
		y_tst = y[i_tst]
		y_pred = classifier.predict(X[i_tst]) # predictions
		for c in range(C):
			i_tst_c = y_tst == c
			if sum(i_tst_c) > 0:
				scores[k, c] = metric(y_tst[i_tst_c], y_pred[i_tst_c])
	return np.nanmean(scores, axis=0) # mean metric per label
