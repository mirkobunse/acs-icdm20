import pandas as pd
import argparse, importlib, os, sys, yaml

# parse the runtime arguments
parser = argparse.ArgumentParser(
	usage='sandbox.plots [options] results_path plot_path',
	add_help=False
)
parser.add_argument('results_path')
parser.add_argument('plot_path')
args = parser.parse_args(sys.argv[-2:]) # only parse the last two arguments
argv = sys.argv[1:-2] # the remaining (previous) options

# read the results file
if not os.path.isfile(args.results_path):
	sys.exit('ERROR: {} does not exist'.format(args.results_path))
results_df = pd.read_csv(args.results_path) # DataFrame of results

# obtain the plotting script (module name derived from directory)
plotter = os.path.basename(os.path.dirname(args.results_path))
module_name = 'sandbox.plots.{}'.format(plotter)
try:
	module = importlib.import_module(module_name)
	module_main = getattr(module, "main") # get a function object
except ModuleNotFoundError:
	print('ERROR: Could not find the module {}'.format(module_name))
except AttributeError:
	print('ERROR: The module {} has no function main'.format(module_name))
else: # i.e. no exception occured above	
	print('\n'.join([
		'-> {}.main(results_df, plot_path, argv)'.format(module_name),
		'   * results_path:  {}'.format(args.results_path),
		'   * plot_path: {}'.format(args.plot_path)
	]))
	module_main(results_df, args.plot_path, argv) # call the plotting function
