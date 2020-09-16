import h5py
import numpy as np
import pandas as pd
import yaml
from os import path
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from urllib.request import urlretrieve

#
# Generic data access API
#
def get_dataset(dataset_id, **kwargs):
	"""Get a data set (X, y, P) identified by its name."""
	return {
	  'spirals': gen_spirals,
	  '3clusters': gen_3clusters,
	  'vertebral': read_vertebral,
	  'yeast': read_yeast,
	  'vehicle': read_vehicle
	}[dataset_id.lower()](**kwargs)

def get_reference(dataset_id):
	"""Return the LaTeX citation command for the data set."""
	return {
	  'spirals': '\\cite{kottke2016probabilistic}',
	  '3clusters': '\\cite{kottke2016probabilistic}',
	  'vertebral': '\\cite{dua2017uci}',
	  'yeast': '\\cite{dua2017uci}',
	  'vehicle': '\\cite{dua2017uci}'
	}[dataset_id.lower()]

def label_counts(y, C=None):
	"""Count the number of occurences of each of the C labels in y"""
	if C is None:
		C = np.max(y) + 1
	lc = np.zeros(C).astype(int) # initialize with zeros
	l, c = np.unique(y, return_counts=True) # labels and counts
	lc[l] = c # make sure missing classes are counted as zeros
	return lc

#
# Read data from CSV
#
def read_vehicle(path="data/vehicle.csv", **kwargs):
	"""See read_from_csv(path, **kwargs)

	Integer values in y encode the labels 'bus' (0), 'van' (1), 'saab' (2), and
	'opel' (3). For more information, see the UCI website.

	https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
	"""
	return read_from_csv(path, **kwargs)

def read_vertebral(path="data/vertebral_column.csv", **kwargs):
	"""See read_from_csv(path, **kwargs)

	Integer values in y encode the labels 'Normal' (0), 'Hernia' (1), and
	'Spondylolisthesis' (2). For more information, see the UCI website.

	http://archive.ics.uci.edu/ml/datasets/vertebral+column
	"""
	return read_from_csv(path, **kwargs)

def read_yeast(path="data/yeast.csv", ignore_small_classes=True, ignore_constant_features=True, **kwargs):
	"""See read_from_csv(path, **kwargs)

	Integer values in y encode the labels 'CYT' (0), 'NUC' (1), 'ME1' (2),
	'ME2' (3), and 'ME3' (4). For more information, see the UCI website.

	http://archive.ics.uci.edu/ml/datasets/yeast
	"""
	X, y, P = read_from_csv(path, **kwargs)
	if ignore_small_classes:
		i = np.logical_and(y != 2, y != 3)
		X, y, P = X[i], y[i], P[i] # select examples from the other labels
		y[y == 4] = 2 # recode labels
	if ignore_constant_features:
		X = X[:, [0, 1, 2, 3, 6, 7]] # ignore almost-constant features 4 and 5
	return X, y, P

def read_from_csv(path, n_samples=None, shuffle=True, verbose=0, **kwargs):
	"""Read a tuple (X, y, _) from a prepared CSV file in which the last column
	contains the integer labels.
	"""
	if len(kwargs) > 0 and verbose > 0:
		print('WARNING: read_from_csv ignores kwargs {}'.format(', '.join(kwargs.keys())))
	X = np.loadtxt(path, delimiter=',')
	if n_samples is None:
		n_samples = len(X)
	else:
		n_samples = min(n_samples, len(X))
	if shuffle:
		np.random.shuffle(X) # shuffle X along the first axis => shuffle instances
	y = X[:,-1].astype(int) # the last column stores the label
	X = np.delete(X, -1, 1) # remove this last column
	P = np.zeros((n_samples,0)) # no parameters available
	return (X[0:n_samples], y[0:n_samples], P)

#
# Synthetic data generation
#
def gen_3clusters(n_samples=600, shuffle=True, config_path="config/data/3clusters.yml", **kwargs):
	"""Generate the synthetic data set (X, y, _) with three clusters."""
	with open(config_path, 'r') as file:
		conf = yaml.safe_load(file) # read the configuration
	conf.update(dict(kwargs)) # overwrite any property configured by the file
	mean = list(map(lambda x: eval(x), conf['mean'])) # mean vector of each cluster
	cov  = list(map(lambda x: eval(x), conf['cov']))
	prop = np.array(conf['proportion'])
	n_samples_per_class = np.ceil(n_samples * prop / prop.sum()).astype(int)
	X = np.concatenate(tuple(map(
	  lambda tup: np.random.multivariate_normal(tup[0], tup[1], tup[2]),
	  zip(mean, cov, n_samples_per_class)
	)))
	y = np.repeat(np.arange(3), n_samples_per_class)
	if shuffle:
		i_perm = np.random.permutation(len(X))
		X = X[i_perm]
		y = y[i_perm]
	P = np.zeros((n_samples,0)) # no parameters available
	return (X[0:n_samples], y[0:n_samples], P)

def gen_spirals(n_samples=1200, shuffle=True, config_path="config/data/spirals.yml", **kwargs):
	"""Generate the synthetic data set (X, y, _) with two spirals and one cluster."""
	with open(config_path, 'r') as file:
		conf = yaml.safe_load(file) # read the configuration
	conf.update(dict(kwargs)) # overwrite any property configured by the file
	prop = np.array(conf['proportion'])
	n_samples_per_class = np.ceil(n_samples * prop / prop.sum()).astype(int)
	X_cluster = np.random.multivariate_normal(
	  eval(conf['cluster_mean']),
	  eval(conf['cluster_cov']),
	  n_samples_per_class[0]
	) # generate the cluster
	theta = tuple(map(
	  lambda tup: np.random.uniform(tup[0]['min'], tup[0]['max'], tup[1]),
	  zip(conf['theta'], n_samples_per_class[1:])
	))
	r = tuple(map(
	  lambda tup: tup[0] + tup[1] * tup[2],
	  zip(conf['a'], conf['b'], theta)
	))
	X_spirals = np.stack(list(map(
	  lambda tup: tup[0] * np.array([ np.cos(tup[1]), np.sin(tup[1]) ]),
	  zip(np.concatenate(theta), np.concatenate(r))
	))) # map to cartesian coordinates
	X = np.concatenate((X_cluster, X_spirals))
	y = np.repeat(np.arange(3), n_samples_per_class)
	X[y==2] *= -1 # TODO add a scale parameter to achieve this effect
	# X[y==2] = np.apply_along_axis(lambda x: x + eval(conf['mean'][1]), 0, X[y==2])
	if shuffle:
		i_perm = np.random.permutation(len(X))
		X = X[i_perm]
		y = y[i_perm]
	P = np.zeros((n_samples,0)) # no parameters available
	return (X[0:n_samples], y[0:n_samples], P)

#
# Sampling
#
def sample_uniformly(y, **kwargs):
	"""See sample_proportions(y, p=np.ones(np.max(y)+1), **kwargs)"""
	return sample_proportions(y, p=np.ones(np.max(y)+1), **kwargs)

def sample_proportionally(y, **kwargs):
	"""See sample_proportions(y, p=np.unique(y, return_counts=True)[1], **kwargs)"""
	return sample_proportions(y, p=np.unique(y, return_counts=True)[1], **kwargs)

def sample_proportions(y, p, max_n_samples=None, min_n_samples=None, shuffle_before=True, shuffle_afterwards=True):
	"""Sample indices of y so that each unique value of y has the given proportion.

	By default, the indices are shuffled before and after the sampling. If not shuffled
	afterwards, they are returned in increasing order of unique values in y.
	"""
	N_s = __n_samples_per_class(y, p, max_n_samples=max_n_samples, min_n_samples=min_n_samples)

	i_all = np.arange(len(y)) # all indices of y
	if shuffle_before:
		np.random.shuffle(i_all)

	i_sel = [] # selected indices
	for label in np.arange(len(p)):
		i_sel.extend(i_all[y[i_all] == label][:N_s[label]])
	i_sel = np.array(i_sel) # continue with a numpy array instead of a regular list
	if shuffle_afterwards:
		np.random.shuffle(i_sel)
	return i_sel

def __n_samples_per_class(y, p, max_n_samples=None, min_n_samples=None):
	"""Determine, how many samples to acquire from y according to the proportions p"""
	if max_n_samples is None:
		max_n_samples = len(y) # at most the entire pool is selected
	if min_n_samples is None:
		min_n_samples = 1 # at least one sample is selected
	if len(y) < min_n_samples:
		raise ValueError('Can not sample {} examples from a pool of size {}'.format(
			min_n_samples,
			len(y)
		))

	# check the proportions argument p
	p = np.array(p) # ensure that p is a numpy array
	N_c = label_counts(y, C=len(p)) # number of occurences of each label
	if not np.all(np.isfinite(p)):
		raise ValueError('p={} is not finite'.format(p))
	if np.all(p == 0):
		raise ValueError('Can not sample with p={}'.format(p))
	if p[N_c > 0].sum() == 0:
		raise ValueError('Can not sample with p={} if N_c={}'.format(
			p, N_c
		)) # only missing classes have a non-zero proportion

	p = p / p.sum() # the desired share of each class
	N = max(
		min_n_samples,
		min(max_n_samples, int(np.min(N_c[p != 0] / p[p != 0])))
	) # the number of samples to acquire in total

	# set up an initial acquisition plan
	p_s = np.rint(p * N) / N # round to the next integer (may not sum up to one)
	while np.any(p_s * N > N_c):
		i_not_enough = p_s * N >= N_c # boolean mask for all classes completely selected
		i_enough = np.logical_not(i_not_enough) # boolean mask for the others
		p_s[i_not_enough] = N_c[i_not_enough] / N # select all samples if there aren't enough
		p[i_not_enough] = p_s[i_not_enough] # also update the desired shares p
		if p_s[i_enough].sum() > 0:
			p_s[i_enough] = p_s[i_enough] / p_s[i_enough].sum() * (1 - np.nansum(p_s[i_not_enough]))
		if p[i_enough].sum() > 0:
			p[i_enough] = p[i_enough] / p[i_enough].sum() * (1 - np.nansum(p[i_not_enough]))
		p_s = np.rint(p_s * N) / N

	# correct the initial acquisition plan
	for _ in range(int(np.abs(np.sum(np.rint(p_s * N)) - N)) + 1):
		if np.sum(np.rint(p_s * N)) == N:
			break # that's the condition we want to achieve
		if np.sum(np.rint(p_s * N)) > N:
			p_e = p_s - p # error of the integer acquisition plan
			p_e[p_s * N <= 0] = -np.inf # do not remove samples you do not have
			m_e = np.argwhere(p_e == np.nanmax(p_e)).flatten() # all samples that are most superfluous
			p_s[np.random.choice(m_e)] -= 1/N # remove one random sample of these
		else:
			p_e = p - p_s # this p_e is the other way round than the above one
			p_e[p_s * N >= N_c] = -np.inf # do not add samples you do not have
			m_e = np.argwhere(p_e == np.nanmax(p_e)).flatten() # all samples that are most urgently needed
			p_s[np.random.choice(m_e)] += 1/N # add one random sample of these

	return np.minimum(np.rint(p_s * N).astype(int), N_c) # the final acquisition plan
