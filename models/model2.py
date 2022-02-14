import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.max_pool2d(x1, 2, 2)
        x2 = F.relu(self.conv2(x))
        x = F.max_pool2d(x2, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x2, x1