import argparse
import torch
import numpy as np
from data.dataset import get_strange_symbol_cv_loaders
from nn.models import options, get_model
from operations import train, validate

"""
Small program to validate a neural network using crossvalidation

"""

parser = argparse.ArgumentParser()

parser.add_argument('nn', type=str, help='the network to operate with: ' + str(options))
parser.add_argument('k', type=int, help='the number of folds to use')
parser.add_argument('--learning_rate', type=float, default=0.05, help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.85, help='the momentum')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
parser.add_argument('--epochs', type=int, default=100, help='the epochs to train')
parser.add_argument('--epslon', type=float, default=0.001, help='the convergence criteria')
parser.add_argument('--seed', type=int, help='the seed to consider in random numbers generation for reproducibility')

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    loaders = get_strange_symbol_cv_loaders(batch_size=args.batch_size, k=args.k)
    avg_train_loss = 0
    avg_train_acc = 0
    avg_val_loss = 0
    avg_val_acc = 0

    figure_id = 0

    for k, loaders in enumerate(loaders):
        print('[CROSSVALIDATION] for', args.nn, '- step', k)
        print()

        net = get_model(args.nn)
        print(net)

        print()

        train_loader, val_loader = loaders
        train_stats = train(net, train_loader, args.learning_rate, args.momentum, epochs=args.epochs, epslon=args.epslon)

        avg_train_loss += train_stats[0]
        avg_train_acc += train_stats[1]

        print('[TRAINING] Final loss in this step', train_stats[0])
        print('[TRAINING] Final acc in this step', train_stats[1])

        val_stats = validate(net, val_loader)

        avg_val_loss += val_stats[0]
        avg_val_acc += val_stats[1]

        print('[VALIDATION] Final loss in this step', val_stats[0])
        print('[VALIDATION] Final acc in this step', val_stats[1])

        print()

    print('[CROSSVALIDATION] Results')
    print('[TRAINING] Avg loss for the model is', avg_train_loss / args.k)
    print('[TRAINING] Avg acc for the model is', avg_train_acc / args.k)
    print('[VALIDATION] Avg loss for the model is', avg_val_loss / args.k)
    print('[VALIDATION] Avg acc for the model is', avg_val_acc / args.k)
