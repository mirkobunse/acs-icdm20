import numpy as np
import pandas as pd
import csv, tqdm
from sandbox import data # import from this project

# conduct the experiment configured by the given dict
def main(config, results_path):

	# prepare a results DataFrame to be filled
	dtypes = {
	  'dataset': str,
	  'n_classes': int,
	  'n_samples': int,
	  'n_features': int,
	  'nca_components': int,
	  'n_parameter_groups': int,
	  'pY': str
	} # p0-pN are floats
	results = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)

	# iterate over all data sets
	for data_config in tqdm.tqdm(config['datasets'], ncols=60):

		# read or generate the data
		dataset_id = data_config['id']
		X, y, P = data.get_dataset(dataset_id)

		n_samples = X.shape[0]
		n_classes = np.max(y) + 1
		pY = '{{$({})$}}'.format(np.array2string(
			data.label_counts(y, n_classes) / len(y),
			separator=', ',
			formatter={'float_kind': lambda x: ("%.2f" % x)[1:]}
		)[1:-1]) # LaTeX string representation

		# hack for artificial data sets
		if dataset_id in ['3clusters', 'spirals', 'bars']:
			n_samples = -1 # unlimited samples
			pY = '{$(\\frac{1}{3}, \\frac{1}{3}, \\frac{1}{3})$}'

		# collect information
		dataset_info = {
		  'dataset': ' '.join([dataset_id, data.get_reference(dataset_id)]),
		  'n_classes': n_classes,
		  'n_samples': n_samples,
		  'n_features': X.shape[1],
		  'nca_components': data_config['nca_components'],
		  'n_parameter_groups': np.max(P) + 1 if P is not None and P.size > 0 else -1,
		  'pY': pY
		}
		for label, count in enumerate(data.label_counts(y)):
			dataset_info['p{}'.format(label)] = count / X.shape[0] # proportion of each class
		results = results.append(dataset_info, ignore_index = True)

	# store the results
	print(results.iloc[:, :9]) # show results before storing them
	results.to_csv(results_path, index=False)
	print('Wrote results to {}'.format(results_path))
