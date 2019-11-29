import os
import csv
import logging
from datetime import datetime

import torch
import numpy as np

import nn.net1
import nn.net2
from nn.models import LeNet5_15, NetSuggested, Linear
from data.cv_dataset import get_cv_datasets, cv_datasets_to_dataloaders
from data.dataset import get_strange_symbols_train_dataset
from operations import train, validate, classify


# hardcoded for now
BATCH_SIZE = 128
device = 'cuda'
lr = 0.05
momentum = 0.85
epslon = 1e-5
epochs = 100

class ModelSelect:
    MSEC_DIR = 'model-select'
    CV_STATS_FILE = 'cv_stats.txt'
    MODEL_FILE_TEMPLATE = 'k{}.pt'

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


    def cross_validate_model(self, model, k, patience, model_dir):
        cv_dss = get_cv_datasets(self.original_dataset, k)
        loaders = cv_datasets_to_dataloaders(cv_dss, BATCH_SIZE, 2)

        # save first weights to reinitialize model after each fold
        model_init_params_path = os.path.join(model_dir, 'init_params.pt')
        model.eval()
        torch.save(model.state_dict(), model_init_params_path)

        avg_train_loss = 0
        avg_train_acc = 0
        avg_val_loss = 0
        avg_val_acc = 0
        
        with open(os.path.join(model_dir, "cv.txt"), 'w') as val_file:
            logging.info("Starting Cross-Validation in {}.".format(model_dir))
            
            for ki, (train_loader, val_loader) in enumerate(loaders, 0):
                logging.info("Cross-Validation k {}".format(ki))

                model.load_state_dict(torch.load(model_init_params_path))

                checkpoint = os.path.join(model_dir, "checkpoint_k{}".format(ki))
                train_stats = train(
                    model, train_loader, lr, momentum, epochs=epochs, 
                    epslon=epslon, device=device, checkpoint=checkpoint)

                train_losses, train_accs = train_stats[2:]
                log_train(train_losses, train_accs, 
                          os.path.join(model_dir, "stats_k{}.csv".format(ki)))            

                avg_train_loss += train_stats[0]
                avg_train_acc += train_stats[1]

                val_loss, val_acc = validate(model, val_loader, device=device)
                avg_val_loss += val_loss
                avg_val_acc += val_acc

                val_file.write("k {} val_loss {} val_acc {}\n".format(
                    ki, val_loss, val_acc))

                print('[TRAINING] Final loss', train_stats[0])
                print('[TRAINING] Final acc', train_stats[1])

                print('[VALIDATION] Final loss', val_loss)
                print('[VALIDATION] Final acc', val_acc)

                print()

            val_file.write(
                "avg_train_loss {} avg_train_acc {} avg_val_loss {} avg_val_acc {}".format(
                    avg_train_loss/k, avg_train_acc/k, avg_val_loss/k, avg_val_acc/k))

        print('[CROSSVALIDATION - TRAINING] Avg loss for the model is', avg_train_loss / k)
        print('[CROSSVALIDATION - TRAINING] Avg acc for the model is', avg_train_acc / k)
        print('[CROSSVALIDATION - VALIDATION] Avg loss for the model is', avg_val_loss / k)
        print('[CROSSVALIDATION - VALIDATION] Avg acc for the model is', avg_val_acc / k)

        print()


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
                self.models[model_name], k=k, patience=patience, 
                model_dir=model_dir)


def log_train(train_losses, train_accs, path):
    fieldnames = ['epoch', 'train loss', 'train acc']
    with open(path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch, stats in enumerate(zip(train_losses, train_accs), 0):
            writer.writerow({
                    'epoch': epoch,
                    'train loss': stats[0],
                    'train acc': stats[1]
                })


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # models = {
    #     'lenet5': LeNet5_15(use_dropout=False, use_batchnorm=False),
    #     'lenet5_dropout': LeNet5_15(use_dropout=True, use_batchnorm=False),
    #     'lenet5_bn': LeNet5_15(use_dropout=False, use_batchnorm=True),
    #     'lenet5_bn_dropout': LeNet5_15(use_dropout=True, use_batchnorm=True)
    # }

    models = {
        'lenet5': LeNet5_15(use_dropout=False, use_batchnorm=False),
        'lenet5_dropout': LeNet5_15(use_dropout=True, use_batchnorm=False),
        'lenet5_bn': LeNet5_15(use_dropout=False, use_batchnorm=True),
        'lenet5_bn_dropout': LeNet5_15(use_dropout=True, use_batchnorm=True),
        'net': NetSuggested(use_dropout=False, use_batchnorm=False),
        'net_dropout': NetSuggested(use_dropout=True, use_batchnorm=False),
        'net_bn': NetSuggested(use_dropout=False, use_batchnorm=True),
        'net_bn_dropout': NetSuggested(use_dropout=True, use_batchnorm=True),
        'linear': Linear()
    }

    ds = get_strange_symbols_train_dataset()

    model_select = ModelSelect(models, ds)
    model_select.search_best_model(k=10, patience=5)
