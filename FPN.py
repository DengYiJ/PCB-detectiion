import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramid, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolution layers for feature filtering
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, kernel_size=1, bias=False) for i in range(len(in_channels))
        ])

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, features):
        # features: List of feature maps from different stages
        # Assuming features are in the order of F_out1, F_out2, F_out3, F_out4, F_out5
        # where F_out1 is the highest resolution and F_out5 is the lowest resolution

        # Apply convolution to each feature map
        conv_features = [conv(features[i]) for i, conv in enumerate(self.conv_layers)]

        # Initialize the top-down feature fusion
        P5 = conv_features[-1]  # Lowest resolution feature
        P4 = self.upsample(P5) + conv_features[-2]
        P3 = self.upsample(P4) + conv_features[-3]
        P2 = self.upsample(P3) + conv_features[-4]
        P1 = self.upsample(P2) + conv_features[-5]

        return P1, P2, P3, P4, P5