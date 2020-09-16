import time
import numpy as np
import pandas as pd
import hashlib, itertools, os, sys, tqdm, traceback, yaml
from multiprocessing import Pool
from sandbox import data, methods, experiments
from sklearn.model_selection import RepeatedKFold
from itertools import chain


# conduct the experiment configured by the given dict
def main(config, results_path):
    start = time.time()
    np.random.seed(config['seed'])
    trial_seeds = np.random.randint(0, 2 ** 32 - 1, size=config['n_trials'])

    # setting-up grid search
    dict_product = lambda a: list(dict(zip(a.keys(), x)) for x in itertools.product(*a.values()))
    parameter_grid_space = []
    for parameter_grid in config['hyperparameter_space']:
        parameter_grid_space += dict_product(parameter_grid)  # collects product of parameter-grid

    for dataset_config in config['datasets']:
        dataset_config['id'] = [ dataset_config['id'] ] # cast this property to a singleton list

    # prepare arguments for the concurrent execution of all experiments
    argument_tuples = list(itertools.product(
        list(chain.from_iterable(map(dict_product, config['datasets']))),
        parameter_grid_space,
        list(range(config['n_trials'])),
        [trial_seeds],
        [config]
    ))  # cartesian product of configurations and trial indices

    # start all experiments concurrently (by default)
    tqdm_kwargs = { 'total': len(argument_tuples), 'disable': config['verbose']>0, 'ncols': 60 }
    if not config['sequential']:
        print(
            'Running {} experiments ({} trials, {} data sets and {} hyperparameter configurations) on {} cores'.format(
                len(argument_tuples),
                config['n_trials'],
                len(config['datasets']),
                len(parameter_grid_space),
                os.cpu_count()
            ))
        with Pool() as pool:
            pd \
                .concat(tqdm.tqdm(pool.imap_unordered(catch_conduct, argument_tuples), **tqdm_kwargs), sort=False) \
                .to_csv(results_path, index=False)
            print('Wrote results to {}'.format(results_path))
            print('runtime: {} sec.'.format(time.time() - start))

    else:
        print(
            'Running {} experiments ({} trials, {} data sets, {} hyperparameter configurations * {}) sequentially'.format(
                len(argument_tuples),
                config['n_trials'],
                len(config['datasets']),
                len(parameter_grid_space)
            ))
        pd \
            .concat(tqdm.tqdm(map(catch_conduct, argument_tuples), **tqdm_kwargs), sort=False) \
            .to_csv(results_path, index=False)
        print('Wrote results to {}'.format(results_path))
        print('runtime: {} sec.'.format(time.time() - start))


# wrap conduct_experiment(...) with error handling
def catch_conduct(args):
    data_config, hyperparameters, trial, trial_seeds, config = args  # un-pack the tuple of arguments
    try:
        return conduct_experiment(
            data_config['id'],
            hyperparameters,
            data_config['nca_components'],
            trial,
            trial_seeds,
            config
        )
    except Exception as any_exception:
        print('\nERROR: \'{}\' during trial {} on {} and {} components with hyperparameter: {} {}\n'.format(
            any_exception,
            trial,
            data_config['id'],
            data_config['nca_components'],
            hyperparameters,
            traceback.format_exc()
        ))
        return pd.DataFrame()  # must return an empty frame


# conduct one configuration defined by a tuple of input arguments
def conduct_experiment(dataset_id, hyperparameters, nca_components, trial, trial_seeds, config):
    np.random.seed(trial_seeds[trial])  # seeding only affects the current process

    trial_msg = "trial {} on {} and {} components with {}".format(trial, dataset_id, nca_components, hyperparameters)  # name of trial used in logging
    X, y, n_classes, i_tst, i_rem = experiments.get_and_split_data(
        dataset_id, trial_msg, config['proportional_test_set'], config['frac_test'], split_pool=False
    )  # obtain the data and split it into a test set, a remaining pool, and a training set

    if config['verbose'] > 1:
        print('DEBUG: md5 checksum of X_tst with N_c={} is {} in {}'.format(
            data.label_counts(y[i_tst], n_classes), hashlib.md5(X[i_tst]).hexdigest(), trial_msg
        ))

    # perform the NeighborhoodComponentsAnalysis
    if X.shape[1] <= nca_components:
        if config['verbose'] > 0:
            print('WARNING: Only found {} features, skipping {}'.format(X.shape[1], trial_msg))
        return pd.DataFrame()  # return an empty frame

    if config['classifier']['parameters'] is not None:
        config['classifier']['parameters'].update(hyperparameters)
    else:
        config['classifier']['parameters'] = hyperparameters

    pipeline = methods.make_pipeline(config['classifier'], C=n_classes, nca_components=nca_components)

    # evaluate the classifier
    rkf = RepeatedKFold(
        n_splits=config['rkf_n_splits'],
        n_repeats=config['rkf_n_repeats'],
        random_state=np.random.randint(0, 2 ** 32 - 1)
    )
    results = pd.DataFrame()  # empty DataFrame to be filled
    for k, (k_trn, k_tst) in enumerate(rkf.split(i_rem)):
        total_error = methods.evaluate_error(X[k_trn], y[k_trn], X[k_tst], y[k_tst], classifier=pipeline)
        results = results.append({
            'trial': trial,
            'dataset': dataset_id,
            'k': k,
            **hyperparameters,
            'nca_components': nca_components if nca_components > 0 else X.shape[1] if nca_components == 0 else X.shape[1] + 1,
            'error': total_error
        }, ignore_index=True)
    if config['verbose'] > 0:
        print('INFO: Finished {}'.format(trial_msg))
    return results  # return the local results DataFrame
