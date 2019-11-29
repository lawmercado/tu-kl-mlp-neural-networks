import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_15(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(LeNet5_15, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)


        # uses empty Sequential Modules to avoid lots of ifs in forward
        self.dropout = nn.Sequential()
        if use_dropout:
            self.droupout = nn.Dropout()

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

        x = self.dropout(x.view(-1, 16 * 4 * 4))

        x = self.dropout(self.batchn2(F.relu(self.fc1(x))))
        x = F.relu(self.batchn3(self.fc2(x)))
        x = self.fc3(x)
        return x