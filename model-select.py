import os
import logging
from datetime import datetime

MSEC_DIR = 'model-select'

def search_best_model(models, original_dataset, k=10, patience=5, logs_dir='./logs'):
	"""
	Use Cross-Validation to select the best model from the given ones.


	"""
	logging.basicConfig(level=logging.DEBUG)

	# create path to save runs if the dont exist
	msec_runs_dir = os.path.join(logs_dir, MSEC_DIR)
	if not os.path.exists(msec_runs_dir):
		os.makedirs(msec_runs_dir)
		logging.info("Created model selection directory in {}".format(
			msec_runs_dir))

	# create log directory for this run
	run_name = datetime.now().strftime('%d_%m_%y__%H_%M_%S')
	run_dir = os.path.join(msec_runs_dir, run_name)
	os.mkdir(run_dir)
	logging.info("Created model selection directory for this run in {}".format(
		run_dir))


if __name__ == "__main__":
	search_best_model(None, None)