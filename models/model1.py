import torch.nn as nn
import torch.nn.functional as F
class Model1(nn.Module):
    def __init__(self, image_size, image_channel, class_num):
        super().__init__()
        stride = 2
        kernel = 5
        padding = 2
        output_padding = 1
        num_filters = 16
        # (image_channel, image_size, image_size) (1, 28, 28)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = image_channel, out_channels = num_filters, kernel_size = kernel, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = num_filters),
            nn.ReLU()
        )
        # (num_filters, (image_size - kernel)//2 + 2) (16, 13, 13)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = num_filters, out_channels = 2*num_filters, kernel_size = kernel, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 2*num_filters),
            nn.ReLU()
        )
        # (2 * num_filters,  ((image_size - kernel)//2 -1)//2) + 1 )  (32, 6, 6)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 2*num_filters, out_channels = num_filters,kernel_size = kernel, stride = 2, dilation =1, padding =1, output_padding = 1),
            nn.BatchNorm2d(num_features = num_filters),
            nn.ReLU()
        )
        # (num_filters, (image_size - kernel)//2 -1)//2) * 2 + 4) (16, 14, 14)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = num_filters, out_channels = class_num, kernel_size = kernel, stride = 2, dilation = 1, padding = 1, output_padding = 1),
            nn.BatchNorm2d(num_features = class_num),
            nn.ReLU()
        )
        # (num_filters, (image_size - kernel)//2 -1)//2) * 4 + 10 (10,30,30)
        self.linear = nn.Sequential(
            nn.Linear(in_features = class_num * (( ((image_size - kernel)// 2- 1)// 2) * 4+ 10)**2, out_features = 60),
            nn.ReLU(),
            nn.Linear(in_features = 60, out_features = class_num),
        )
    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = output.view(output.size(0),-1)
        output = self.linear(output)
        return output