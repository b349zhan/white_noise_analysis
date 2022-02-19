import torch.nn as nn
import torch.nn.functional as F
import torch
class Model2(nn.Module):
    def __init__(self, image_size, image_channel, num_classes):
        super(Model2, self).__init__()
        stride = 2
        kernel = 3
        padding = 2
        output_padding = 1
        num_filters = 16
        # (image_channel, image_size, image_size) (1, 28, 28)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = image_channel, out_channels = num_filters, kernel_size = kernel, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = num_filters),
            nn.ReLU()
        )
        # (num_filters, (image_size - kernel)//2 + 2) (16, 14, 14)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = num_filters, out_channels = 2*num_filters, kernel_size = kernel, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 2*num_filters),
            nn.ReLU()
        )
        # (2 * num_filters, ((image_size - kernel)//2 + 1)//2) + 1 )   (32, 7, 7)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 2*num_filters, out_channels = num_filters,kernel_size = kernel, stride = 2, dilation =1, padding =1, output_padding = 1),
            nn.BatchNorm2d(num_features = num_filters),
            nn.ReLU()
        )
        # (num_filters, ((image_size - kernel)//2 + 1)//2) * 2 + 2 ) (16, 14, 14)
        
        # (2 * num_filters, ((image_size - kernel)//2 + 1)//2) * 2 + 2 ) (16, 14, 14)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 2 * num_filters, out_channels = num_classes, kernel_size = kernel, stride = 2, dilation = 1, padding = 1, output_padding = 1),
            nn.BatchNorm2d(num_features = num_classes),
            nn.ReLU()
        )

        # (num_classes, ((image_size - kernel)//2 + 1)//2) * 4 + 4 )  (32, 28, 28)
        self.linear = nn.Sequential(
            nn.Linear(in_features = (num_classes+image_channel) * \
                      ((((image_size - kernel)//2 + 1)//2) * 4 + 4 )**2, out_features = 60),
            nn.ReLU(),
            nn.Linear(in_features = 60, out_features = num_classes),
        )
        
    def forward(self, x):
        output1 = self.block1(x)
        output2 = self.block2(output1)
        output3 = self.block3(output2)
        input_block4 = torch.cat((output1,output3),1)
        output4 = self.block4(input_block4)
        output = torch.cat((x, output4),1)
        output = output.view(output.size(0),-1)
        output = self.linear(output)
        return output