import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.layer_one = options.layers[0]
        self.network()

    def network(self):
        self.linear1 = nn.Linear(11, self.layer_one)
        self.linear2 = nn.Linear(self.layer_one, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Conv_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network()

    def network(self):
        self.conv1 = nn.Conv2d(3, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class Deep_QNet(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.layer_one = options.layers[0]
        self.layer_two = options.layers[1]
        self.layer_three = options.layers[2]
        self.network()

    def network(self):
        self.linear1 = nn.Linear(11, self.layer_one)
        self.linear2 = nn.Linear(self.layer_one, self.layer_two)
        self.linear3 = nn.Linear(self.layer_two, self.layer_three)
        self.linear4 = nn.Linear(self.layer_three, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
