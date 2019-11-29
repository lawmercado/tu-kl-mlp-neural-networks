import argparse
import torch
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import get_strange_symbol_loader_with_validation, get_strange_symbol_cv_loaders
from data.dataset import get_strange_symbols_test_data
from nn.net1 import Net
from operations import train, validate, test
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.05, help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.85, help='the momentum')
parser.add_argument('--patience', type=int, default=5, help='the patience factor for the early stopping')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
parser.add_argument('--epochs', type=float, default=100, help='the epochs to train')
parser.add_argument('--epslon', type=float, default=0.001, help='the convergence criteria')
parser.add_argument('--crossvalidate', type=int,
                    help='whether should the model be validated with crossvalidation k parameter or not')
parser.add_argument('--holdout', type=float, help='whether should the model be validated with the holdout '
                                                  '(with percentage for the validation) method or not')
parser.add_argument('--confusion_matrix', action='store_true', help='whether should the confusion matrix be generated')
parser.add_argument('--tops_and_bottoms', action='store_true', help='whether should be presented the tops and bottom'
                                                                    ' classifications')
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

            train_stats = train(
                net, train_loader, args.lr, args.momentum, epochs=args.epochs, epslon=args.epslon,
                device=device)

            avg_train_loss += train_stats[0]
            avg_train_acc += train_stats[1]

            print('[TRAINING] Final loss', train_stats[0])
            print('[TRAINING] Final acc', train_stats[1])

            val_stats = validate(net, val_loader)

            avg_val_loss += val_stats[0]
            avg_val_acc += val_stats[1]

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
        loaders = get_strange_symbol_loader_with_validation(batch_size=args.batch_size, validation_split=args.holdout)
        avg_train_loss = 0
        avg_train_acc = 0
        avg_val_loss = 0

        net = Net()
        train_loader, val_loader = loaders

        train_stats = train(
            net, train_loader, args.lr, args.momentum, epochs=args.epochs, 
            epslon=args.epslon, device=device)

        print('[TRAINING] Final loss', train_stats[0])
        print('[TRAINING] Final acc', train_stats[1])

        avg_val_loss, avg_val_acc = validate(net, val_loader)

        print('[VALIDATION] Final loss', avg_val_loss)
        print('[VALIDATION] Final acc', avg_val_acc)

        print()

        predictions = []
        labels = []
        image_tensors = []
        output_tensors = []

        for val_data in val_loader:
            val_imgs, val_labels = val_data

            output = test(net, val_imgs)
            prediction = output.argmax(dim=1)

            output_tensors.append(softmax(output, dim=1))
            image_tensors.append(val_imgs)
            predictions = predictions + prediction.tolist()
            labels = labels + val_labels.tolist()

        if args.tops_and_bottoms:
            outputs = torch.cat(output_tensors, dim=0)
            imgs = torch.cat(image_tensors, dim=0)

            certainties = []

            for i in range(len(imgs)):
                certainties.append(outputs[i][labels[i]].item())

            indexes = np.argsort(np.array(certainties))

            top_10 = indexes[-10:]
            bottom_10 = indexes[:10]

            top_10_imgs = imgs[top_10[0]].squeeze(0).numpy()
            bottom_10_imgs = imgs[bottom_10[0]].squeeze(0).numpy()

            for i in range(1, len(top_10)):
                top_img = imgs[top_10[i]].squeeze(0).numpy()
                top_10_imgs = np.hstack((top_10_imgs, top_img))
                bottom_img = imgs[bottom_10[i]].squeeze(0).numpy()
                bottom_10_imgs = np.hstack((bottom_10_imgs, bottom_img))

            plt.imsave('plots/top_imgs.png', top_10_imgs, cmap='Greys')

            print('Generated top 10 certainties images')
            print('Certainties:', [certainties[i] for i in top_10])

            plt.imsave('plots/bottom_imgs.png', bottom_10_imgs, cmap='Greys')

            print('Generated bottom 10 certainties images')
            print('Certainties:', [certainties[i] for i in bottom_10])

        if args.confusion_matrix:
            cm = confusion_matrix(np.array(labels), np.array(predictions))

            print('Confusion matrix: ')
            print(cm)

        if args.plot:
            epochs = np.linspace(0, len(train_stats[2]), len(train_stats[2]))

            f1 = plt.figure(2)
            plt.plot(epochs, train_stats[2], color='coral', label='training')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.legend()
            plt.title('Loss x epochs')

            f2 = plt.figure(3)
            plt.plot(epochs, train_stats[3], color='coral', label='training')
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