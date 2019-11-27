import os
import logging
from datetime import datetime



#TODO: using a class could be better since logging needs to be global across the functions

class ModelSelect:
	MSEC_DIR = 'model-select'

	def __init__(self, models, original_dataset, logs_dir='./logs'):
		self.models = models
		self.original_dataset = original_dataset
		self.logs_dir = logs_dir

		self.run_dir = None
		self.initialize_logging_dirs()


	def initialize_logging_dirs(self):
		logging.basicConfig(level=logging.DEBUG)

		# create path to save runs if the dont exist
		msec_runs_dir = os.path.join(self.logs_dir, self.MSEC_DIR)
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

		self.run_dir = run_dir


	def cross_validate_model(self, model, k, patience):
		pass


	def search_best_model(self, k=10, patience=5):
		"""
		Use Cross-Validation to select the best model from the given ones.


		"""
		for model_name in self.models:
			model_dir = os.path.join(self.run_dir, model_name)
			os.mkdir(model_dir)
			logging.info(
				"Model selection artifacts for {} will be saved to {}".format(
					model_name, model_dir))

			self.cross_validate_model(
				self.models[model_name], k=k, patience=patience)


if __name__ == "__main__":
	models = {
		'model1': None,
		'model2': None
	}

	model_select = ModelSelect(models, None)
	model_select.search_best_model(k=10, patience=5)