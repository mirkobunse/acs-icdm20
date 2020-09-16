import numpy as np
from sandbox import data, methods # import from this project

def get_and_split_data(dataset_id, trial_msg, proportional_test_set, n_tst, n_trn=None, split_pool=True, n_tst_per_class=True, n_trn_per_class=True):
	"""Get the data configured by data_config and split it into a test set,
	a remaining pool, and optionally a training set."""
	X, y, _ = data.get_dataset(dataset_id) # read or generate the data
	C = np.max(y) + 1

	# split the data into a test set (i_tst) and a remaining pool (i_rem)
	if isinstance(n_tst, float) and n_tst < 1:
		n_tst = int(n_tst * len(y)) # n_tst is a fraction; round to the floor
	elif n_tst_per_class:
		n_tst = C * n_tst # n_tst is a class-wise absolute count; else, it is the total count already
	if proportional_test_set:
		i_tst = data.sample_proportionally(y, max_n_samples=n_tst)
	else:
		i_tst = data.sample_uniformly(y, max_n_samples=n_tst)
	if len(i_tst) < n_tst:
		print('WARNING: Test set only has N_c={} in {}'.format(data.label_counts(y[i_tst], C), trial_msg))
	i_rem = np.setdiff1d(np.arange(len(y)), i_tst) # remaining indices

	# split a training set (i_trn) from the pool
	if split_pool and n_trn is None:
		raise ValueError('n_trn must not be None if split_pool is True')
	elif split_pool:
		if n_trn_per_class:
			n_trn_per_class = n_trn # per-class count
			n_trn = C * n_trn # total count
		else:
			n_trn_per_class = n_trn / C
		p_trn = np.minimum(
			np.ones(C, dtype=int) * n_trn_per_class,
			data.label_counts(y[i_rem], C)
		) # feasible class proportions in the remaining data
		i_trn = i_rem[data.sample_proportions(y[i_rem], p=p_trn, max_n_samples=p_trn.sum())]
		i_rem = np.setdiff1d(i_rem, i_trn) # still remaining indices
		if len(i_trn) < n_trn:
			print('WARNING: Training set only has N_c={} (remaining: {}) in {}'.format(
				data.label_counts(y[i_trn], C), data.label_counts(y[i_rem], C), trial_msg
			))
		return X, y, C, i_tst, i_rem, i_trn
	else:
		return X, y, C, i_tst, i_rem

#
# Simplex iteration
#
def simplex_iterator(scale, C, boundary=True):
	if C == 2:
		return __binary_simplex_iterator(scale, boundary)
	elif C == 3:
		return __ternary_simplex_iterator(scale, boundary)
	elif C == 4:
		return __quaternary_simplex_iterator(scale, boundary)
	else:
		raise ValueError('C must be in [2, 3, 4]')

# from github.com/marcharper/python-ternary
def __ternary_simplex_iterator(scale, boundary=True):
	start = 0
	if not boundary:
		start = 1
	for i in range(start, scale + (1 - start)):
		for j in range(start, scale + (1 - start) - i):
			k = scale - i - j
			yield (i, j, k)

# same for binary classification
def __binary_simplex_iterator(scale, boundary=True):
	start = 0
	if not boundary:
		start = 1
	for i in range(start, scale + (1 - start)):
		j = scale - i
		yield (i, j)

# is 'quaternary' even a word?
def __quaternary_simplex_iterator(scale, boundary=True):
	start = 0
	if not boundary:
		start = 1
	for i in range(start, scale + (1 - start)):
		for j in range(start, scale + (1 - start) - i):
			for k in range(start, scale + (1 - start) - i - j):
				l = scale - i - j -k
				yield (i, j, k, l)
