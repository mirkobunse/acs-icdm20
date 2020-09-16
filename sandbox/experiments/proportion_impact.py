import numpy as np
import pandas as pd
import hashlib, itertools, os, sys, tqdm, traceback, yaml
from multiprocessing import Pool
from sandbox import data, methods, experiments # import from this project
from sandbox.methods import acs # fix eval statements

# conduct the experiment configured by the given dict
def main(config, results_path):
	np.random.seed(config['seed'])
	trial_seeds = np.random.randint(0, 2**32-1, size = config['n_trials'])

	# FIXME expand proportions on the simplex

	# prepare arguments for the concurrent execution of all experiments
	argument_tuples = list(itertools.product(
	  config['datasets'],
	  list(range(config['n_trials'])),
	  [ trial_seeds ],
	  [ config ]
	)) # cartesian product of configurations and trial indices

	# start all experiments concurrently (by default)
	tqdm_kwargs = { 'total': len(argument_tuples), 'disable': config['verbose']>0, 'ncols': 60 }
	if not config['sequential']:
		print('Running {} experiments ({} trials * {} data sets) on {} cores'.format(
		  len(argument_tuples),
		  config['n_trials'],
		  len(config['datasets']),
		  os.cpu_count()
		))
		with Pool() as pool:
			pd \
			  .concat(tqdm.tqdm(pool.imap_unordered(catch_conduct, argument_tuples), **tqdm_kwargs)) \
			  .to_csv(results_path, index = False)
			print('Wrote results to {}'.format(results_path))
	else:
		print('Running {} experiments ({} trials * {} data sets) sequentially'.format(
		  len(argument_tuples),
		  config['n_trials'],
		  len(config['datasets'])
		))
		pd \
		  .concat(tqdm.tqdm(map(catch_conduct, argument_tuples), **tqdm_kwargs)) \
		  .to_csv(results_path, index = False)
		print('Wrote results to {}'.format(results_path))


# wrap conduct_experiment(...) with error handling
def catch_conduct(args):
	data_config, trial, trial_seeds, config = args # un-pack the tuple of arguments
	try:
		return conduct_experiment(
			data_config,
			trial,
			trial_seeds,
			config
		)
	except Exception as any_exception:
		print('\nERROR: \'{}\' during trial {} on {}\n{}'.format(
			any_exception,
			trial,
			data_config['id'],
			traceback.format_exc()
		))
		return pd.DataFrame() # must return an empty frame

# conduct one configuration defined by a tuple of input arguments
def conduct_experiment(data_config, trial, trial_seeds, config):
	np.random.seed(trial_seeds[trial]) # seeding only affects the current process

	# read or generate the data
	dataset_id = data_config['id']
	trial_msg = "trial {} on {}".format(trial, dataset_id) # name of trial used in logging
	X, y, n_classes, i_tst, i_rem = experiments.get_and_split_data(
		dataset_id, trial_msg, config['proportional_test_set'], config['frac_test'], split_pool=False
	) # obtain the data and split it into a test set, a remaining pool, and a training set
	simplex_scale = data_config['simplex_scale']
	pY_tst = data.label_counts(y[i_tst], C=n_classes) / len(i_tst) # assumed true proportions

	# prepare the evaluation
	pipeline_args = {'C': n_classes, 'nca_components': data_config['nca_components'], 'input_dim': X.shape[1]}

	#sets classifier hyper-parameters
	try:
		if data_config['classifier_parameters'] is not None:
			config['classifier']['parameters'].update(data_config['classifier_parameters'])
	except KeyError:
		pass

	clf = methods.make_pipeline(config['classifier'], **pipeline_args) # preprocessing + classification
	results = pd.DataFrame() # empty DataFrame to be filled

	if config['verbose'] > 1:
		print('DEBUG: md5 checksum of X_tst with N_c={} is {} in {}'.format(
			data.label_counts(y[i_tst], n_classes), hashlib.md5(X[i_tst]).hexdigest(), trial_msg
		))

	# increase the total number of training samples and try all proportions on the simplex
	for k, simplex_scale_multiple in enumerate(data_config['simplex_scale_multiples']):
		N_trn = int(simplex_scale * simplex_scale_multiple) # total number of training samples
		for N_c in experiments.simplex_iterator(simplex_scale, n_classes, boundary=False):

			# sample training data
			i_trn = i_rem[data.sample_proportions(y[i_rem], N_c, max_n_samples=N_trn, shuffle_before=False, shuffle_afterwards=False)]
			label_counts = data.label_counts(y[i_trn], C=n_classes)
			if label_counts.sum() != N_trn:
				if config['verbose'] > 1:
					print('DEBUG: Can\'t sample {} from {} data with N_c={}'.format(
						N_trn * np.array(N_c) / np.sum(N_c), dataset_id, data.label_counts(y[i_rem], C=n_classes)
					)) # not a big issue, just some missing rows
				continue # other proportions may still be available

			# evaluate
			total_error = methods.evaluate_error(X[i_trn], y[i_trn], X[i_tst], y[i_tst], classifier=clf)
			pY_trn = label_counts / label_counts.sum() # training set proportions
			dY = np.sum(pY_tst * np.log(pY_tst / pY_trn)) # KL divergence wrt Y
			results = results.append({
			  'trial': trial,
			  'k': k+1,
			  'N': len(i_trn),
			  'p0': pY_trn[0],
			  'p1': pY_trn[1],
			  'p2': pY_trn[2] if len(pY_trn) > 2 else -1,
			  'p3': pY_trn[3] if len(pY_trn) > 3 else -1,
			  'dataset': dataset_id,
			  'dY': dY,
			  'error': total_error
			}, ignore_index = True)

	if config['verbose'] > 0:
		print('..Finished {}'.format(trial_msg))
	return results # return the local results DataFrame
