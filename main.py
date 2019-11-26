import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import get_strange_symbol_loader, get_strange_symbol_cv_loaders, get_strange_symbols_test_data
from nn.net1 import Net
from operations import train, test
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.05, help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.85, help='the momentum')
parser.add_argument('--patience', type=int, default=5, help='the patience factor for the early stopping')
parser.add_argument('--batch_size', type=float, default=128, help='the batch size')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--crossvalidate', type=int,
                    help='whether should the model be validated with crossvalidation k parameter or not')
parser.add_argument('--holdout', type=float, help='whether should the model be validated with the holdout '
                                                  '(with percentage for the validation) method or not')
parser.add_argument('--confusion_matrix', action='store_true', help='whether should the confusion matrix be generated')
parser.add_argument('--plot', action='store_true', help='whether the results should be plotted')
parser.add_argument('--seed', type=int, help='the seed to consider in random numbers generation for reproducibility')
parser.add_argument('--cuda', action='store_true', help='use cuda if available')

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # should make cuda runs deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cpu")
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
print("Using {} device.".format(device))


if __name__ == '__main__':

    if args.crossvalidate:
        loaders = get_strange_symbol_cv_loaders(batch_size=args.batch_size, k=args.crossvalidate)
        avg_train_loss = 0
        avg_train_acc = 0
        avg_val_loss = 0
        avg_val_acc = 0

        figure_id = 0

        for k, loaders in enumerate(loaders):
            print('- CV step', k, '-')
            net = Net()
            train_loader, val_loader = loaders
            train_stats, val_stats = train(net, train_loader, val_loader,
                                           args.lr, args.momentum, patience=args.patience,
                                           epochs=args.epochs, device=device)

            avg_train_loss += train_stats[0]
            avg_train_acc += train_stats[1]
            avg_val_loss += val_stats[0]
            avg_val_acc += val_stats[1]

            print('[TRAINING] Final loss', train_stats[0])
            print('[TRAINING] Final acc', train_stats[1])
            print('[VALIDATION] Final loss', val_stats[0])
            print('[VALIDATION] Final acc', val_stats[1])

            print()

        print('[CROSSVALIDATION - TRAINING] Avg loss for the model is', avg_train_loss / args.crossvalidate)
        print('[CROSSVALIDATION - TRAINING] Avg acc for the model is', avg_train_acc / args.crossvalidate)
        print('[CROSSVALIDATION - VALIDATION] Avg loss for the model is', avg_val_loss / args.crossvalidate)
        print('[CROSSVALIDATION - VALIDATION] Avg acc for the model is', avg_val_acc / args.crossvalidate)

        print()

    elif args.holdout:
        print('- Holdout -')
        loaders = get_strange_symbol_loader(batch_size=args.batch_size, validation_split=args.holdout)
        avg_train_loss = 0
        avg_train_acc = 0
        avg_val_loss = 0

        net = Net()
        train_loader, val_loader = loaders
        train_stats, val_stats = train(net, train_loader, val_loader,
                                       args.lr, args.momentum, patience=args.patience,
                                       epochs=args.epochs)

        print('[TRAINING] Final loss', train_stats[0])
        print('[TRAINING] Final acc', train_stats[1])
        print('[VALIDATION] Final loss', val_stats[0])
        print('[VALIDATION] Final acc', val_stats[1])

        print()

        if args.confusion_matrix:
            predictions = []
            labels = []

            for val_data in val_loader:
                val_imgs, val_labels = val_data

                predictions = predictions + test(net, val_imgs).tolist()
                labels = labels + val_labels.tolist()

            cm = confusion_matrix(np.array(labels), np.array(predictions))

            print('Confusion matrix: ')
            print(cm)

        if args.plot:
            epochs = np.linspace(0, len(train_stats[2]), len(train_stats[2]))

            f1 = plt.figure(0)
            plt.plot(epochs, train_stats[2], color='coral', label='training')
            plt.plot(epochs, val_stats[2], color='teal', label='validation')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend()
            plt.title('Loss x epochs')

            f2 = plt.figure(1)
            plt.plot(epochs, train_stats[3], color='coral', label='training')
            plt.plot(epochs, val_stats[3], color='teal', label='validation')
            plt.ylabel('acc')
            plt.xlabel('epochs')
            plt.legend()
            plt.title('Acc x epochs')

    # TODO
    # Now it's up to you to define the network and use the data to train it.
    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the test data using the provided method:
    test_data = get_strange_symbols_test_data()
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt

    plt.show()