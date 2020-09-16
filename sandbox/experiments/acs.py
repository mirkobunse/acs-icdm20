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

	# prepare arguments for the concurrent execution of all experiments
	argument_tuples = list(itertools.product(
	  config['datasets'],
	  config['scorers'],
	  list(range(config['n_trials'])),
	  [ trial_seeds ],
	  [ config ]
	)) # cartesian product of configurations and trial indices

	# start all experiments concurrently (by default)
	tqdm_kwargs = { 'total': len(argument_tuples), 'disable': config['verbose']>0, 'ncols': 60 }
	if not config['sequential']:
		print('Running {} experiments ({} trials * {} data sets * {} scorers) on {} cores'.format(
		  len(argument_tuples),
		  config['n_trials'],
		  len(config['datasets']),
		  len(config['scorers']),
		  os.cpu_count()
		))
		with Pool() as pool:
			pd \
			  .concat(tqdm.tqdm(pool.imap_unordered(catch_conduct, argument_tuples), **tqdm_kwargs)) \
			  .to_csv(results_path, index = False)
			print('Wrote results to {}'.format(results_path))
	else:
		print('Running {} experiments ({} trials * {} data sets * {} scorers) sequentially'.format(
		  len(argument_tuples),
		  config['n_trials'],
		  len(config['datasets']),
		  len(config['scorers'])
		))
		pd \
		  .concat(tqdm.tqdm(map(catch_conduct, argument_tuples), **tqdm_kwargs)) \
		  .to_csv(results_path, index = False)
		print('Wrote results to {}'.format(results_path))


# wrap conduct_experiment(...) with error handling
def catch_conduct(args):
	data_config, scorer_config, trial, trial_seeds, config = args # un-pack the tuple of arguments
	try:
		return conduct_experiment(
			data_config,
			scorer_config,
			trial,
			trial_seeds,
			config
		)
	except Exception as any_exception:
		print('\nERROR: \'{}\' during trial {} on {} and {}\n{}'.format(
			any_exception,
			trial,
			data_config['id'],
			scorer_config['class_name'],
			traceback.format_exc()
		))
		return pd.DataFrame() # must return an empty frame

# conduct one configuration defined by a tuple of input arguments
def conduct_experiment(data_config, scorer_config, trial, trial_seeds, config):
	np.random.seed(trial_seeds[trial]) # seeding only affects the current process

	# basic configuration
	scorer_class = eval(scorer_config['class_name'])
	dataset_id = data_config['id']
	trial_msg = "trial {} on {} and {}".format(trial, dataset_id, scorer_class.__name__) # name of trial used in logging

	data_config['n_rounds'] = len(data_config['n_samples'])-1

	# read data
	X, y, n_classes, i_tst, i_rem, i_trn = experiments.get_and_split_data(
		dataset_id, trial_msg, config['proportional_test_set'], config['frac_test'], data_config['n_samples'][0], n_trn_per_class=False
	) # obtain the data and split it into a test set, a remaining pool, and a training set
	pY_tst = data.label_counts(y[i_tst], C=n_classes) / len(i_tst) # assumed true proportions

	# prepare the scorer
	parameters = scorer_config.get('parameters', {}).copy()
	for key in parameters:
		parameters[key] = eval(str(parameters[key]))
	pipeline_args = {'C': n_classes, 'nca_components': data_config['nca_components'], 'input_dim': X.shape[1]}
	scorer = scorer_class(
	  X[i_trn],
	  y[i_trn],
	  C = n_classes,
	  preprocessing = methods.make_pipeline(**pipeline_args),
	  logging_name = trial_msg,
	  proportions = data.label_counts(y[i_tst], n_classes),
	  **parameters
	)
	
	if config['verbose'] > 1:
		print('DEBUG: md5 checksum of X_tst with N_c={} is {} in {}'.format(
			data.label_counts(y[i_tst], n_classes), hashlib.md5(X[i_tst]).hexdigest(), trial_msg
		))

	# baseline error with the initial training set
	label_counts = data.label_counts(y[i_trn], C=n_classes)
	pY_trn = label_counts / label_counts.sum() # training set proportions
	dY = np.sum(pY_tst * np.log(pY_tst / pY_trn)) # KL divergence wrt Y

	# sets classifier hyper-parameters
	try:
		if data_config['classifier_parameters'] is not None:
			config['classifier']['parameters'].update(data_config['classifier_parameters'])
	except KeyError:
		pass

	clf = methods.make_pipeline(config['classifier'], **pipeline_args) # preprocessing + classification
	classwise_error = methods.evaluate_classwise_error(X[i_trn], y[i_trn], X[i_tst], y[i_tst], C=n_classes, classifier=clf)
	results = pd \
	  .DataFrame() \
	  .append({
	    'trial': trial,
	    'k': 0,
	    'N': len(i_trn),
	    'p0': pY_trn[0],
	    'p1': pY_trn[1],
	    'p2': pY_trn[2] if len(pY_trn) > 2 else -1,
	    'p3': pY_trn[3] if len(pY_trn) > 3 else -1,
	    'dataset': dataset_id,
	    'scorer': scorer_class.__name__,
	    'dY': dY,
	    'error': methods.evaluate_error(X[i_trn], y[i_trn], X[i_tst], y[i_tst], classifier=clf),
	    'error_0': classwise_error[0],
	    'error_1': classwise_error[1],
	    'error_2': classwise_error[2] if len(pY_trn) > 2 else -1,
	    'depleted': np.any(data.label_counts(y[i_rem], C=n_classes) == 0)
	  }, ignore_index = True)

	# active class selection: iteratively select labels to generate examples for
	for k in range(data_config['n_rounds']):

		config['n_samples_per_round'] = data_config['n_samples'][k+1] - data_config['n_samples'][k]

		# distinguish between LabelScores (ACS) and InstanceScores (AL)
		if issubclass(type(scorer), methods.acs.LabelScore):
			scores = scorer.score_labels(config['n_samples_per_round']) # score labels to select the best ones
			i_add = i_rem[methods.acs.select_from_pool(
				y[i_rem],
				scores,
				N = config['n_samples_per_round'],
				log_msg = 'k={}, {}'.format(k+1, trial_msg) if config['verbose'] > 0 else None
			)] # select samples from the pool, according to the given scores

		elif issubclass(type(scorer), methods.al.InstanceScore):
			random_batch = i_rem[np.concatenate(list(map(
			  lambda label: np.argwhere(y[i_rem]==label)[0:min(25, np.sum(y[i_rem]==label))],
			  range(n_classes)
			)))].reshape(-1)
			scores = scorer.score_instances(X[random_batch])
			i_add = random_batch[np.argsort(scores)[::-1][:config['n_samples_per_round']]]

		else:
			raise TypeError('The scorer must be of type LabelScore or InstanceScore')

		i_trn = np.concatenate((i_trn, i_add)) # add this example to the training data
		i_rem = np.setdiff1d(i_rem, i_add) # remove it from the remaining pool
		scorer.add_data(X[i_add], y[i_add]) # register new data at the scorer

		# evaluate
		label_counts = data.label_counts(y[i_trn], C=n_classes)
		if config['verbose'] > 2:
			print('TRACE: N_c={} (k={}, {})'.format(label_counts, k, trial_msg))
		pY_trn = label_counts / label_counts.sum()
		dY = np.sum(pY_tst * np.log(pY_tst / pY_trn))
		total_error = methods.evaluate_error(X[i_trn], y[i_trn], X[i_tst], y[i_tst], classifier=clf)
		classwise_error = methods.evaluate_classwise_error(X[i_trn], y[i_trn], X[i_tst], y[i_tst], C=n_classes, classifier=clf)
		results = results.append({
		  'trial': trial,
		  'k': k+1,
		  'N': len(i_trn),
		  'p0': pY_trn[0],
		  'p1': pY_trn[1],
		  'p2': pY_trn[2] if len(pY_trn) > 2 else -1,
		  'p3': pY_trn[3] if len(pY_trn) > 3 else -1,
		  'dataset': dataset_id,
		  'scorer': scorer_class.__name__,
		  'dY': dY,
		  'error': total_error,
		  'error_0': classwise_error[0],
		  'error_1': classwise_error[1],
		  'error_2': classwise_error[2] if len(pY_trn) > 2 else -1,
		  'depleted': np.any(data.label_counts(y[i_rem], C=n_classes) == 0)
		}, ignore_index = True)

	if config['verbose'] > 0:
		print('INFO: Finished {} with labels {} and error {}'.format(trial_msg, label_counts, total_error))
	return results # return the local results DataFrame
