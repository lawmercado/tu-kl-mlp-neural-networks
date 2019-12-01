import torch.nn as nn
import torch.nn.functional as F

options = ['basic', 'lenet5']


class BasicNet(nn.Module):
    """
    The neural network suggested in the task description

    """

    def __init__(self):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 15)

    def forward(self, x):
        # Flattens the data for the fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5(nn.Module):
    """
    Our version of the LeNet5 network with some slight changes

    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout2 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)  # here we adapt the output

    def forward(self, x):
        # Convolution layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flattens the data for the fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout2(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_model(name):
    if name not in options:
        raise NotImplementedError('Network not not implemented')

    if name == 'lenet5':
        return LeNet5()

    elif name == 'basic':
        return BasicNet()