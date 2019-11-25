import argparse
import torch
import numpy as np
from data.dataset import get_strange_symbol_loader, get_strange_symbols_test_data
from nn.net import Net
from torch import nn
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.05, help="the learning rate")
parser.add_argument("--momentum", type=float, default=0.85, help="the momentum")
parser.add_argument("--batch_size", type=float, default=128, help="the batch size")
parser.add_argument("--seed", type=int, help="the seed to consider in random numbers generation")

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    train_loader = get_strange_symbol_loader(batch_size=args.batch_size)

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):

        avg_loss = 0
        avg_acc = 0

        for i, data in enumerate(train_loader):
            imgs, labels = data  # data is a batch of samples, split into an image tensor and label tensor

            optimizer.zero_grad()  # zero the gradient buffers

            # output is a bi dimensional tensor (imgs.shape[0]x15)
            output = net(imgs)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()  # Does the weight update

            avg_loss += loss.item()

            predictions = output.argmax(dim=1)
            avg_acc += (predictions == labels).sum().item()/len(labels)

        print('Avg loss in epoch', epoch, 'is', avg_loss/len(train_loader))
        print('Avg accuracy in epoch', epoch, 'is', avg_acc / len(train_loader))
        print('--')

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
