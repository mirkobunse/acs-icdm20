import numpy as np
import pandas as pd
import itertools, os, scipy.stats, sys, traceback, yaml
from multiprocessing import Pool
from sandbox import data, methods # import from this project
from tqdm import tqdm, trange

# conduct the experiment configured by the given dict
def main(config, results_path):
	np.random.seed(config['seed'])

	# prepare arguments for the concurrent execution of all experiments
	for index_dataset, data_config in enumerate(config['datasets']):
		data_config['index'] = index_dataset
	argument_tuples = list(itertools.product(
	  config['datasets'],
	  [ config ]
	)) # cartesian product of configurations and trial indices

	# start all experiments concurrently (by default) or in sequence
	if not config['sequential']:
		print('Running {} experiments on {} cores'.format(len(argument_tuples), os.cpu_count()))
		with Pool() as pool:
			pd \
			  .concat(pool.imap_unordered(catch_conduct, argument_tuples)) \
			  .to_csv(results_path, index = False)
			print('Wrote results to {}'.format(results_path))
	else:
		print('Running {} experiments sequentially'.format(len(argument_tuples)))
		pd \
		  .concat(map(catch_conduct, argument_tuples)) \
		  .to_csv(results_path, index = False)
		print('Wrote results to {}'.format(results_path))

# wrap conduct_experiment(...) with error handling
def catch_conduct(args):
	data_config, config = args # unpack the argument tuple
	try:
		return conduct_experiment(data_config, config)
	except Exception as any_exception:
		print('\nERROR: \'{}\' during experiment on {} data\n{}'.format(
			any_exception,
			data_config['id'],
			traceback.format_exc()
		))
		return pd.DataFrame() # must return an empty frame

# conduct one configuration defined by a tuple of input arguments
def conduct_experiment(data_config, config):
	np.random.seed(config['seed'])

	# read or generate the data
	dataset_id = data_config['id']
	X, y, _ = data.get_dataset(dataset_id)
	C = np.max(y) + 1 # the number of classes
	nY = data.label_counts(y, C) # number of samples in each class

	trial_msg = "experiment on {} data".format(dataset_id) # name of trial used in logging
	if config['verbose'] > 1:
		tqdm.write('INFO: Starting {}'.format(trial_msg))

	# assume the true distribution was characterized by KDEs
	pXY = [ scipy.stats.gaussian_kde(X[y == c].T) for c in range(C) ]
	pY = nY / nY.sum()
	log_pY = np.log(pY)
	tqdm.write('P(Y) of {} data: {}'.format(dataset_id, pY))

	# linearly interpolate between pY and inv_pY
	results = pd.DataFrame() # empty DataFrame
	for k in trange(config['n_iterations'], position=data_config['index'], desc='{:20s}'.format(dataset_id), disable=config['verbose']>0):
		for c in range(C):

			# create the class proportions in iteration k
			k_pC = config['min_pC'] + (1-2*config['min_pC']) * (k / (config['n_iterations'] - 1)) # P(Y=c)
			k_pY = np.insert(np.delete(pY, c) / np.delete(pY, c).sum() * (1-k_pC), c, k_pC) # maintain true distribution of other classes

			k_nY = np.rint(k_pY * config['n_samples']).astype(int) # number of samples per class
			k_pY = k_nY / k_nY.sum() # update to account for rounding errors
			k_log_pY = np.log(k_pY)
			dY = np.sum(k_pY * np.log(k_pY / pY)) # KL divergence wrt Y

			# repeat each iteration
			for r in range(config['n_repetitions']):

				# resample wrt k_pY
				r_X = np.concatenate([ pXY[c].resample(k_nY[c]).T for c in range(C) ])

				# KL divergence wrt X from p(X | Y) and p(Y)
				r_hat_log_pXY = np.concatenate([ pXY[c].logpdf(r_X.T).reshape(-1,1) for c in range(C) ], axis=1)
				r_hat_pX = np.sum(np.exp(r_hat_log_pXY + k_log_pY.reshape(1,-1)), axis=1) # sum(r_hat_pXY[:,i] * k_pY[i])
				r_pX = np.sum(np.exp(r_hat_log_pXY + log_pY.reshape(1,-1)), axis=1) # true probability
				r_hat_pX /= r_hat_pX.sum() # normalize
				r_pX /= r_pX.sum()
				dX = np.sum(r_hat_pX * (np.log(r_hat_pX) - np.log(r_pX)))

				# # KL divergence of p(Y | X)
				# # caution: this implementation does not produce dYX = dY - dX
				# r_hat_pYX = np.exp(r_hat_log_pXY + k_log_pY.reshape(1,-1)) / r_hat_pX.reshape(-1,1)
				# r_pYX = np.exp(r_hat_log_pXY + log_pY.reshape(1,-1)) / r_pX.reshape(-1,1)
				# r_hat_pYX /= r_hat_pYX.sum(axis=1, keepdims=True) # normalize
				# r_pYX /= r_pYX.sum(axis=1, keepdims=True)
				# nonzero_r_pYX = np.all(r_pYX > 0, axis=1) # KL divergence ignores zero probability events
				# r_hat_pYX = r_hat_pYX[nonzero_r_pYX, :]
				# r_hat_pX = r_hat_pX[nonzero_r_pYX]
				# r_pYX = r_pYX[nonzero_r_pYX, :]
				# dYX = np.sum(r_hat_pX * np.sum(r_hat_pYX * (np.log(r_hat_pYX) - np.log(r_pYX)), axis=1))
				dYX = -1

				# KL divergence of p(X | Y) is assumed to be zero

				if config['verbose'] > 0:
					tqdm.write('[{:2d}, P(Y={:d})={:.3f}] dY={:.3e} -> dX={:.3e} on {} data with k_pY={}'.format(
						k, c, k_pC, dY, dX, dataset_id, k_pY
					))

				results = results.append({
				  'dataset': dataset_id,
				  'k': k,
				  'c': c,
				  'pC': k_pC,
				  'r': r,
				  'dY': dY,
				  'dYX': dYX,
				  'dX': dX,
				}, ignore_index = True)

	if config['verbose'] > 0:
		tqdm.write('INFO: Finished {}'.format(trial_msg))
	return results # return the local results DataFrame
