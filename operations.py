import torch
from torch import nn
from torch import optim
from torch.nn.functional import softmax
import numpy as np
from os import remove


def train(net, train_loader, lr, momentum, epochs, epslon):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []

    final_train_acc = 0
    final_train_loss = 0
    previous_error = 0

    for epoch in range(epochs):
        print('[TRAINING] Epoch', epoch)
        net.train()

        avg_train_loss = 0
        avg_train_acc = 0

        for i, train_data in enumerate(train_loader):
            imgs, labels = train_data  # data is a batch of samples, split into an image tensor and label tensor

            optimizer.zero_grad()  # zero the gradient buffers

            output = net(imgs)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()  # Does the weight update

            avg_train_loss += loss.item()
            predictions = output.argmax(dim=1)
            avg_train_acc += (predictions == labels).sum().item() / len(labels)

        final_train_loss = avg_train_loss / len(train_loader)
        final_train_acc = avg_train_acc / len(train_loader)

        print('[TRAINING] Avg loss is', final_train_loss)
        print('[TRAINING] Avg accuracy is', final_train_acc)
        print()

        train_losses.append(final_train_loss)
        train_accs.append(final_train_acc)

        if np.absolute(previous_error - final_train_loss) <= epslon:
            print('[TRAINING] Convergence criteria reached... Stopping training.')
            print()

            break
        else:
            previous_error = final_train_loss

    return final_train_loss, final_train_acc, train_losses, train_accs


def validate(net, val_loader):
    net.eval()

    criterion = nn.CrossEntropyLoss()
    avg_val_loss = 0
    avg_val_acc = 0

    with torch.no_grad():
        for j, val_data in enumerate(val_loader):
            val_imgs, val_labels = val_data
            output = net(val_imgs)
            loss = criterion(output, val_labels)

            avg_val_loss += loss.item()
            predictions = output.argmax(dim=1)
            avg_val_acc += (predictions == val_labels).sum().item() / len(val_labels)

    return avg_val_loss / len(val_loader), avg_val_acc / len(val_loader)


def train_with_early_stopping(net, train_loader, val_loader, lr, momentum, patience, epochs=100):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    epoch = 0
    smallest_val_loss = float('inf')
    final_val_acc = 0
    final_train_acc = 0
    final_train_loss = 0
    greater_val_error_count = 0
    best_val_epoch = 0

    while greater_val_error_count < patience and epoch < epochs:
        print('[TRAINING] Epoch', epoch)
        net.train()

        avg_train_loss = 0
        avg_train_acc = 0

        for i, train_data in enumerate(train_loader):
            imgs, labels = train_data  # data is a batch of samples, split into an image tensor and label tensor

            optimizer.zero_grad()  # zero the gradient buffers

            # output is a bi dimensional tensor (imgs.shape[0]x15)
            output = net(imgs)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()  # Does the weight update

            avg_train_loss += loss.item()
            predictions = output.argmax(dim=1)
            avg_train_acc += (predictions == labels).sum().item() / len(labels)

        avg_val_loss, avg_val_acc = validate(net, val_loader)

        print('[TRAINING] Avg loss is', avg_train_loss / len(train_loader))
        print('[TRAINING] Avg accuracy is', avg_train_acc / len(train_loader))
        print('[VALIDATION] Avg loss is', avg_val_loss)
        print('[VALIDATION] Avg accuracy is', avg_val_acc)

        print()

        train_losses.append(avg_train_loss / len(train_loader))
        train_accs.append(avg_train_acc / len(train_loader))
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        if avg_val_loss < smallest_val_loss:
            smallest_val_loss = avg_val_loss
            final_val_acc = avg_val_acc
            final_train_acc = avg_train_acc / len(train_loader)
            final_train_loss = avg_train_loss / len(train_loader)

            torch.save(net.state_dict(), '.tmp_nn.pt')
            greater_val_error_count = 0
            best_val_epoch = epoch
        else:
            greater_val_error_count += 1

        epoch += 1

    net.load_state_dict(torch.load('.tmp_nn.pt'))
    net.eval()

    remove('.tmp_nn.pt')

    print('[TRAINING] Epoch with the least error in the validation set is', best_val_epoch)

    return (final_train_loss, final_train_acc, train_losses, train_accs), \
           (smallest_val_loss, final_val_acc, val_losses, val_accs)


def test(net, test_data, normalize=False):
    net.eval()
    with torch.no_grad():
        output = net(test_data)

        if normalize:
            output = softmax(output, dim=1)

        return output


def classify(net, test_data):
    output = test(net, test_data)
    predictions = output.argmax(dim=1)

    return predictions


def get_tops_and_bottoms_images_indexes(outputs, labels, n=10):
    certainties = []

    for i in range(len(outputs)):
        certainties.append(outputs[i][labels[i]].item())

    indexes = np.argsort(np.array(certainties))

    tops = indexes[-n:]
    bottoms = indexes[:n]

    return tops, bottoms
