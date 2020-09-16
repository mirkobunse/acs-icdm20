import argparse, importlib, os, sys, warnings, yaml

# parse the runtime arguments
parser = argparse.ArgumentParser('sandbox.experiments', add_help=False)
parser.add_argument('config_path')
parser.add_argument('results_path')
args = parser.parse_args()

# ignore some particularly annoying warnings
warnings.filterwarnings('ignore', message='Variables are collinear.')

# read the configuration file
if not os.path.isfile(args.config_path):
	sys.exit('ERROR: {} does not exist'.format(args.config_path))
with open(args.config_path, 'r') as file:
	config = yaml.safe_load(file)

# obtain the experiment script (module name derived from directory)
experiment = os.path.basename(os.path.dirname(args.config_path))
module_name = 'sandbox.experiments.{}'.format(experiment)
try:
	module = importlib.import_module(module_name)
	module_main = getattr(module, "main") # get a function object
except ModuleNotFoundError:
	print('ERROR: Could not find the module {}'.format(module_name))
except AttributeError:
	print('ERROR: The module {} has no function main'.format(module_name))
else: # i.e. no exception occured above
	print('\n'.join([
		'-> {}.main(config, results_path)'.format(module_name),
		'   * config_path:  {}'.format(args.config_path),
		'   * results_path: {}'.format(args.results_path)
	]))
	module_main(config, args.results_path) # pass the configuration to the experiment
