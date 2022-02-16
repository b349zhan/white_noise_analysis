import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.cnn_model = nn.Sequential(
                 nn.Conv2d(1, 6, kernel_size = 5), #(N, 1, 28, 28) -> (N, 6, 24, 24)
                 nn.Tanh(),
                 nn.AvgPool2d(2, stride = 2), #(N, 6, 24, 24) -> (N, 6, 12, 12)
                 nn.Conv2d(6, 16, kernel_size = 5), #(N, 6, 12, 12) -> (N, 6, 8, 8)
                 nn.Tanh(),
                 nn.AvgPool2d(2, stride = 2)) 
        self.fc_model = nn.Sequential(
                 nn.Linear(256, 120), # (N, 256) -> (N, 120)
                 nn.Tanh(),
                 nn.Linear(120, 84), # (N, 120) -> (N, 84)
                 nn.Tanh(),
                 nn.Linear(84, 10))  # (N, 84)  -> (N, 10)) #10 classes   
    def forward(self, x):      
        x = self.cnn_model(x)     
        x = x.view(x.size(0), -1)     
        x = self.fc_model(x)     
        return x
