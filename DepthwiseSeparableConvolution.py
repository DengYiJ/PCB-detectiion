import torch.nn as nn
import torch
from torchsummary import summary


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm1=nn.BatchNorm2d(in_channels)
        self.norm2=nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.depthwise(x)
        x=self.norm1(x)
        x=self.relu(x)
        x = self.pointwise(x)
        x=self.norm2(x)
        x=self.relu(x)
        return x

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = DepthwiseSeparableConvolution(3, 4, kernel_size=1, padding=0).to(device)
print(summary(conv, input_size=(3, 64, 64)))'''
