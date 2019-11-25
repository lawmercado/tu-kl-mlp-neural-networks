import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.conv3 = nn.Conv2d(12, 24, 2)
        self.batchn1 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 2 * 2, 64)  # 2*2 from image dimension after pooling
        self.batchn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.batchn3 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 15)

    def forward(self, x):
        # Convolution layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.batchn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flattens the data for the fully connected layers
        x = x.view(-1, self.num_flat_features(x))

        # Fully connected layers
        x = self.fc1(x)
        x = self.batchn2(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batchn3(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
