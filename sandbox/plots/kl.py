import pandas as pd
import numpy as np
import argparse, itertools, sys
from sandbox import plots # import from this project

def main(df, plot_path, argv):
	parser = argparse.ArgumentParser('sandbox.plots.kl', add_help=False)
	parser.add_argument('--omit-fix', default=False, action='store_true')
	args = parser.parse_args(argv)

	if not plot_path.endswith('.csv'):
		sys.exit("The output path must end on .csv")
	plot_path_template = plot_path[::-1].replace(
		'.csv'[::-1], '_{}_{}.csv'[::-1], 1
	)[::-1] # replace only the last occurence of '.csv'

	# dX <= dY may not always be satisfied, due to sampling errors
	if not args.omit_fix:
		print('Fixing sampling errors')
		to_fix = df['dX'] > df['dY'] # boolean mask
		fix_values = (df['dX'] + df['dY']) / 2 # mean values
		df.loc[to_fix, 'dX'] = fix_values[to_fix]
		df.loc[to_fix, 'dY'] = fix_values[to_fix]

	df['dY_minus_dX'] = df['dY'] - df['dX']

	# aggregate over repetitions, then flatten the column names
	df = df \
	  .groupby(['dataset', 'k', 'c', 'pC']) \
	  .describe()[list(itertools.product(
	    ['dX', 'dY', 'dYX', 'dY_minus_dX'],
	    ['mean', 'std']
	  ))] \
	  .reset_index()
	df.columns = [ '_'.join(tup).rstrip('_') for tup in df.columns.values ]
	df['dX_var'] = np.power(df['dX_std'], 2) # variance of the error
	df['dY_var'] = np.power(df['dY_std'], 2)
	df['dYX_var'] = np.power(df['dYX_std'], 2)
	df['dY_minus_dX_var'] = np.power(df['dY_minus_dX_std'], 2)

	# store one file for each scorer and dataset
	for group_tuple, group_df in df.groupby(['dataset', 'c']):
		dataset, c = group_tuple # un-pack tuple
		d_path = plot_path_template.format(dataset, int(c)) # replace '{}'s
		print('Writing to {}'.format(d_path))
		group_df \
		  .drop(['dataset', 'c'], axis=1) \
		  .to_csv(d_path, index=False)

	print('Writing all results to {}'.format(plot_path))
	df.to_csv(plot_path, index=False)
