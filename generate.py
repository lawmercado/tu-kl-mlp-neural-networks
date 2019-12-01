import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import get_strange_symbol_loader_with_validation
from nn.models import options, get_model
from operations import train_with_early_stopping

"""
Small program to generate the final neural network

"""

parser = argparse.ArgumentParser()

parser.add_argument('nn', type=str, help='the network to operate with: ' + str(options))
parser.add_argument('s', type=float, help='the split fraction used for the validation set')
parser.add_argument('p', type=int, help='the patience factor used in early stopping')
parser.add_argument('--learning_rate', type=float, default=0.05, help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.85, help='the momentum')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
parser.add_argument('--epochs', type=float, default=100, help='the epochs to train')
parser.add_argument('--seed', type=int, help='the seed to consider in random numbers generation for reproducibility')
parser.add_argument('--plot', action='store_true', help='whether the results should be plotted')

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    print('[GENERATE] model for', args.nn)
    print()

    net = get_model(args.nn)
    print(net)

    print()

    avg_train_loss = 0
    avg_train_acc = 0
    avg_val_loss = 0
    avg_val_acc = 0

    loaders = get_strange_symbol_loader_with_validation(batch_size=args.batch_size, validation_split=args.s)

    train_loader, val_loader = loaders
    train_stats, val_stats = train_with_early_stopping(net, train_loader, val_loader, args.learning_rate, args.momentum,
                                                       args.p, args.epochs)

    print('[TRAINING] Final loss', train_stats[0])
    print('[TRAINING] Final acc', train_stats[1])

    print('[VALIDATION] Final loss', val_stats[0])
    print('[VALIDATION] Final acc', val_stats[1])

    torch.save(net, args.nn + '_nn.pt')

    print()

    if args.plot:
        epochs = np.linspace(0, len(train_stats[2]), len(train_stats[2]))

        f1 = plt.figure(2)
        plt.plot(epochs, train_stats[2], color='coral', label='training')
        plt.plot(epochs, val_stats[2], color='teal', label='validation')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Loss x epochs')

        f2 = plt.figure(3)
        plt.plot(epochs, train_stats[3], color='coral', label='training')
        plt.plot(epochs, val_stats[3], color='teal', label='validation')
        plt.ylabel('acc')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Acc x epochs')

        plt.show()
