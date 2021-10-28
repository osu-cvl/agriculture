import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.conv1 = conv3x3(in_planes=3, out_planes=8)
        self.conv2 = conv3x3(in_planes=8, out_planes=16)
        self.conv3 = conv3x3(in_planes=16, out_planes=16)
        self.max_pool = nn.MaxPool2d((3, 3), 3)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x