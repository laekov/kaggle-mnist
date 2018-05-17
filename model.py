import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(4608, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p = 0.2)
        x = F.relu(self.conv5(x))
        x = F.dropout2d(x, p = 0.2)
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p = 0.5)
        x = x.view((-1, 4608))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

