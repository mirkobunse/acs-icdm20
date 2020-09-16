import pandas as pd
import numpy as np
import argparse, itertools, sys

def main(df, plot_path, argv):
	parser = argparse.ArgumentParser('sandbox.plots.proportion_impact', add_help=False)
	parser.add_argument('-t', '--trial', type=int)
	parser.add_argument('-s', '--scorer')
	parser.add_argument('-a', '--additional') # if the --scorer is not in df
	args = parser.parse_args(argv)

	if not plot_path.endswith('.csv'):
		sys.exit("The output path must end on .csv")
	plot_path_template = plot_path[::-1].replace(
		'.csv'[::-1], '_{}.csv'[::-1], 1
	)[::-1] # replace only the last occurence of '.csv'

	# read additional DataFrame (optional)
	scorer_df = None # default
	if None in [args.additional, args.scorer] and args.additional is not args.scorer:
		raise ValueError('Arguments --scorer and --additional must be given together')
	elif args.additional is not None:
		print('Reading additional results from {}'.format(args.additional))
		scorer_df = pd.read_csv(args.additional) # DataFrame of more results
		scorer_df = scorer_df[scorer_df['scorer'] == args.scorer]
		print('Keeping {} additional entries of {}'.format(
			len(scorer_df),
			args.scorer
		))

	# select a single trial (optional)
	if args.trial is not None:
		df = df[df['trial'] == args.trial]
		print('INFO: Selected trial {} with {} lines'.format(args.trial, len(df)))
	else:
		print('INFO: Read {} lines, using all trials'.format(len(df)))

	for dataset, ddf in df.groupby('dataset'):

		if scorer_df is None:

			# default case, no scorer provided: evaluate proportional sampling
			p = ddf \
			  .groupby(['p0', 'p1', 'p2', 'p3']) \
			  .apply(lambda x: x['dY'].mean()) \
			  .idxmin() # (p0, p1, p2, p3) of proportional sampling (or whatever is closest)
			print('DEBUG: Proportions of {} ~ {}'.format(dataset, p))

			#
			# Determine the rank of the proportional sampling in a single trial and iteration.
			# Note that by sorting wrt dY as a second level we favor proportional sampling.
			# This favor is given to ease interpretation of the resulting plot: The relative
			# rank returned by this function is the percentage of strategies in the simplex
			# which performs strictly better than proportional sampling.
			#
			def proportional_rank(kdf):
				kdf = kdf.sort_values(['error', 'dY']).reset_index(drop=True)
				p_line = kdf[ \
				  (kdf['p0'] == p[0]) & \
				  (kdf['p1'] == p[1]) & \
				  (kdf['p2'] == p[2]) & \
				  (kdf['p3'] == p[3]) \
				] # still a DataFrame, hopefully of length 1
				if len(p_line) == 1:
					p_line = p_line.iloc[0] # everything okay -> cast as a Series
				else:
					return -1 # proportional strategy is missing from kdf
				rank = p_line.name # rank = index in the ordered kdf = the Series name
				return rank / len(kdf) # relative rank

			# determine the ranks (see above)
			ddf = ddf \
			  .groupby(['N', 'trial']) \
			  .apply(proportional_rank)

		else:

			#
			# Determine the rank of the scorer in a single trial and iteration.
			#
			def scorer_rank(kdf):
				N, trial = kdf.name # which trial and iteration are we dealing with?
				s_line = scorer_df[
				  (scorer_df['dataset'] == dataset) &
				  (scorer_df['N'] == N) &
				  (scorer_df['trial'] == trial)
				]
				if len(s_line) == 1:
					s_line = s_line.iloc[0]
				else:
					return -1 # scorer is missing or is ambiguous
				kdf = kdf.append({
				  'error': s_line['error'],
				  'dY': -1, # small hack makes sort_values place the scorer up front
				}, ignore_index=True)
				kdf = kdf.sort_values(['error', 'dY']).reset_index(drop=True)
				kdf_line = kdf[kdf['dY'] == -1] # find the line of the scorer
				if len(kdf_line) == 1:
					kdf_line = kdf_line.iloc[0]
				else:
					return -1
				rank = kdf_line.name
				return rank / len(kdf)

			ddf = ddf \
			  .groupby(['N', 'trial']) \
			  .apply(scorer_rank)

		#  check the results
		if np.sum(ddf == -1) > 0:
			print('WARNING: Broken iterations in {}:\n{}\n'.format(dataset, ddf[ddf == -1]))
			ddf = ddf[ddf > -1] # remove broken iterations

		# compute statistics (mean, std, ...) among all trials
		ddf = ddf \
		  .groupby('N') \
		  .describe() \
		  .reset_index() \
		  .drop(['count'], axis=1)
		ddf.columns = [ c if c=='N' else 'rank_{}'.format(c.rstrip('%')) for c in ddf.columns ]
		ddf['rank_var'] = np.power(ddf['rank_std'], 2)
		ddf['rank_25_error'] = ddf['rank_mean'] - ddf['rank_25'] # relative error used in pgfplots
		ddf['rank_75_error'] = ddf['rank_75'] - ddf['rank_mean']

		# store one file for each dataset
		lc_path = plot_path_template.format(dataset) # replace occurences of {}
		print('INFO: Writing a learning curve of size {} to {}'.format(len(ddf), lc_path))
		ddf.to_csv(lc_path, index=False) # store aggregation in output file

	print('Writing an empty DataFrame to {}'.format(plot_path))
	pd.DataFrame().to_csv(plot_path, index=False) # create an empty file for GNU make to recognize
