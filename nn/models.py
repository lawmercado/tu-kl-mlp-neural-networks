import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_15(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(LeNet5_15, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)


        # uses empty Sequential Modules to avoid lots of ifs in forward
        self.dropout = nn.Sequential()
        if use_dropout:
            self.dropout = nn.Dropout()

        self.batchn1 = nn.Sequential()
        self.batchn2 = nn.Sequential()
        self.batchn3 = nn.Sequential()
        if use_batchnorm:
            self.batchn1 = nn.BatchNorm2d(16)
            self.batchn2 = nn.BatchNorm1d(120)
            self.batchn3 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.batchn1(F.relu(self.conv2(x))))

        x = self.dropout(x.view(-1, 16 * 5 * 5))

        x = self.dropout(self.batchn2(F.relu(self.fc1(x))))
        x = F.relu(self.batchn3(self.fc2(x)))
        x = self.fc3(x)
        return x


class LeNet5_15_Pure(nn.Module):
    def __init__(self):
        super(LeNet5_15_Pure, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_15_BN(nn.Module):
    def __init__(self):
        super(LeNet5_15_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.batchn2 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.batchn3 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.batchn1(F.relu(self.conv2(x))))

        x = x.view(-1, 16 * 5 * 5)

        x = self.batchn2(F.relu(self.fc1(x)))
        x = F.relu(self.batchn3(self.fc2(x)))
        x = self.fc3(x)
        return x


class LeNet5_15_Dropout(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(LeNet5_15_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.dropout1(x.view(-1, 16 * 5 * 5))

        x = self.dropout2(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetSuggested(nn.Module):
    """
    The neural network suggested in the assignment

    """

    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(NetSuggested, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 15)

        self.dropout1 = nn.Sequential()
        self.dropout2 = nn.Sequential()
        if use_dropout:
            self.dropout1 = nn.Dropout()
            self.dropout2 = nn.Dropout()

        self.batchn1 = nn.Sequential()
        self.batchn2 = nn.Sequential()
        if use_batchnorm:
            self.batchn1 = nn.BatchNorm1d(512)
            self.batchn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.dropout1(x.view(-1, self.num_flat_features(x)))
        x = self.dropout2(self.batchn1(F.relu(self.fc1(x))))
        x = self.batchn2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(784, 15)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
