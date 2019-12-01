import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import get_strange_symbol_loader_with_validation
from nn.models import options, get_model
from operations import train, validate, test, get_tops_and_bottoms_images_indexes
from sklearn.metrics import confusion_matrix

"""
Small program to validate a neural network using holdout

"""

parser = argparse.ArgumentParser()

parser.add_argument('nn', type=str, help='the network to operate with: ' + str(options))
parser.add_argument('s', type=float, help='the split fraction used for the validation set')
parser.add_argument('--learning_rate', type=float, default=0.05, help='the learning rate')
parser.add_argument('--momentum', type=float, default=0.85, help='the momentum')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
parser.add_argument('--epochs', type=int, default=100, help='the epochs to train')
parser.add_argument('--epslon', type=float, default=0.0001, help='the convergence criteria')
parser.add_argument('--seed', type=int, help='the seed to consider in random numbers generation for reproducibility')
parser.add_argument('--plot', action='store_true', help='whether the results should be plotted')

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':

    if args.nn not in options:
        raise NotImplementedError('Network not recognized or not implemented')

    print('[HOLDOUT] for', args.nn)
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
    train_stats = train(net, train_loader, args.learning_rate, args.momentum, epochs=args.epochs, epslon=args.epslon)

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

        output = test(net, val_imgs, normalize=True)
        prediction = output.argmax(dim=1)

        output_tensors.append(output)
        image_tensors.append(val_imgs)
        predictions = predictions + prediction.tolist()
        labels = labels + val_labels.tolist()

    cm = confusion_matrix(np.array(labels), np.array(predictions))

    print('[HOLDOUT] Confusion matrix in the validation set', cm)

    if args.plot:
        tops_idx, bottoms_idx = get_tops_and_bottoms_images_indexes(torch.cat(output_tensors, dim=0), labels)
        imgs = torch.cat(image_tensors, dim=0)

        top_imgs = imgs[tops_idx[0]].squeeze(0).numpy()
        bottom_imgs = imgs[bottoms_idx[0]].squeeze(0).numpy()

        for i in range(1, len(tops_idx)):
            top_img = imgs[tops_idx[i]].squeeze(0).numpy()
            top_imgs = np.hstack((top_imgs, top_img))
            bottom_img = imgs[bottoms_idx[i]].squeeze(0).numpy()
            bottom_imgs = np.hstack((bottom_imgs, bottom_img))

        plt.imsave('plots/' + args.nn + 'top_imgs.png', top_imgs, cmap='Greys')
        plt.imsave('plots/' + args.nn + 'bottom_imgs.png', bottom_imgs, cmap='Greys')

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

        plt.show()
