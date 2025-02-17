import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu_


class ConvBN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        print("ConvBN initialized")

    def forward(self,x):
        x = self.conv(x)
        return self.norm(x)

class ResidualBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super().__init__()
        self_layer1=ConvBN(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=1)
        self_layer2=ConvBN(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=1)

    def forward(self,x):
        x=self.layer1(x);
        x=torch.relu(x);
        x=self.layer2(x);
        return x

class ResidualBasicBlockShortcut(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super().__init__()
        self.layer1 = ConvBN(in_channels,out_channels,kernel_size=kernel_size,stride=stride)
        self.layer2 = ConvBN(out_channels,out_channels,kernel_size=kernel_size,stride=1)
        #self.layer3 = ConvBN(out_channels,out_channels,kernel_size=kernel_size,stride=1)
        self.shortcut=ConvBN(in_channels,out_channels,kernel_size=1,stride=2,padding=0)

    def forward(self,x):
        out=self.layer1(x)
        out=torch.relu(out)
        out=self.layer2(out)
        shortcut=self.shortcut(x)
        return x+shortcut

class ResNet(nn.Module):
    def __init__(self,num_classes=10):
       super().__init__()
       self.l1 = ConvBN(3, 64, kernel_size=7, stride=2, padding=3)
       self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       self.l2a = ResidualBasicBlock(64, 64)
       self.l2b = ResidualBasicBlock(64, 64)
       self.l2c = ResidualBasicBlock(64, 64)
       self.l3a = ResidualBasicBlockShortcut(64, 128)
       self.l3b = ResidualBasicBlock(128, 128)
       self.l3c = ResidualBasicBlock(128, 128)
       self.l3d = ResidualBasicBlock(128, 128)
       self.l4a = ResidualBasicBlockShortcut(128, 256)
       self.l4b = ResidualBasicBlock(256, 256)
       self.l4c = ResidualBasicBlock(256, 256)
       self.l4d = ResidualBasicBlock(256, 256)
       self.l4e = ResidualBasicBlock(256, 256)
       self.l4f = ResidualBasicBlock(256, 256)
       self.l5a = ResidualBasicBlockShortcut(256, 512)
       self.l5b = ResidualBasicBlock(512, 512)
       self.l5c = ResidualBasicBlock(512, 512)
       self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
       self.flatten = nn.Flatten()
       self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.max_pool(x)
        x = self.l2a(x)
        x = self.l2b(x)
        x = self.l2c(x)
        x = self.l3a(x)
        x = self.l3b(x)
        x = self.l3c(x)
        x = self.l3d(x)
        x = self.l4a(x)
        x = self.l4b(x)
        x = self.l4c(x)
        x = self.l4d(x)
        x = self.l4e(x)
        x = self.l4f(x)
        x = self.l5a(x)
        x = self.l5b(x)
        x = self.l5c(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x