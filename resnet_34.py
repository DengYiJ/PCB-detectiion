# from torch import nn
# from torchvision.models import ResNet34_Weights
# import torchvision.models as models
# # 18、34版本残差块
# '''
# @:param
# 		input_channels: 输入通道数
# 		num_channels: 输出通道数
# 		use_1x1conv: 是否使用 1x1conv (使用就说明是卷积映射残差块, 要变换尺寸)
# 		strides: 步长
# '''
# class Residual_primary(nn.Module):
# 	def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
# 		# 默认 use_1x1conv=False, strides=1
# 		super(Residual_primary, self).__init__()
# 		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
# 		self.bn1 = nn.BatchNorm2d(num_channels)  # BN层
# 		self.relu1 = nn.ReLU(inplace=True)  # Inplace ReLU to save memory
# 		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
# 		self.bn2 = nn.BatchNorm2d(num_channels)
# 		# 如果用了 1x1conv 就说明是卷积映射残差块, 要变换尺寸
# 		if use_1x1conv:
# 			self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
# 			self.bn3 = nn.BatchNorm2d(num_channels)
# 		else:
# 			self.conv3 = None
#
# 	def forward(self, X):
# 		Y = self.conv1(X)
# 		Y = self.bn1(Y)
# 		Y = self.relu1(Y)
# 		Y = self.conv2(Y)
# 		Y = self.bn2(Y)
# 		if self.conv3:
# 			X = self.conv3(X)
# 			X = self.bn3(X)  # If using 1x1 conv, also apply BN
# 		Y += X
# 		Y = nn.ReLU(inplace=True)(Y)  # Optionally, you can move this ReLU outside the Residual class
# 		return Y
#
# # 初始卷积层和池化层
# b1 = nn.Sequential(
# 	nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Removed bias since we have BN, 输入通道为3表示彩色图像
# 	nn.BatchNorm2d(64),
# 	nn.ReLU(inplace=True),
# 	nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# )
#
# # 大残差结构
# '''
# @:param
# 		input_channels: 输入通道数
# 		num_channels: 输出通道数
# 		num_residuals: 残差块的个数
# 		first_block: 是否是第一个大残差结构
# @:return
# 		nn.Sequential(*blk): 大残差结构
# '''
# def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
# 	blk = []
# 	for i in range(num_residuals):
# 		stride = 2 if i == 0 and not first_block else 1  # 从第二个大残差结构开始, 结构中的第一个残差块一般都会尺寸减半, 即 stride=2
# 		use_1x1conv = i == 0 and not first_block  # use_1x1conv = False/True, 从第二个大残差结构开始, 结构中的第一个残差块都是卷积映射残差块
# 		if i == 0:
# 			blk.append(Residual_primary(in_channels, out_channels, use_1x1conv=use_1x1conv, strides=stride))
# 		else:
# 			blk.append(Residual_primary(out_channels, out_channels, strides=stride))
# 	return nn.Sequential(*blk)
#
#
# # ResNet-18
# # def resnet18(num_classes, in_channels=3):
# # 	# net = nn.Sequential(b1)
# # 	net = nn.Sequential(
# # 		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
# # 		nn.BatchNorm2d(64),
# # 		nn.ReLU(inplace=True),
# # 		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# # 	)
# # 	net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
# # 	net.add_module("resnet_block2", resnet_block(64, 128, 2))
# # 	net.add_module("resnet_block3", resnet_block(128, 256, 2))
# # 	net.add_module("resnet_block4", resnet_block(256, 512, 2))
# # 	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
# # 	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
# # 	# Optionally, initialize weights here (e.g., nn.init.kaiming_normal_(net[0].conv1.weight, mode='fan_out', nonlinearity='relu'))
# # 	return net
#
#
# # ResNet-34
# class ResNet34_FeatureExtractor(nn.Module):
# 	def __init__(self, in_channels=3,pretrained=True):
# 		super().__init__()
# 		# 使用 weights 参数加载预训练权重，这里使用最新的推荐方式
# 		weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
# 		resnet34 = models.resnet34(weights=weights)
# 		# 初始卷积层
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(inplace=True),
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 		)
# 		# 四个残差块阶段
# 		self.block1 = resnet_block(64, 64, 3, first_block=True)
# 		self.block2 = resnet_block(64, 128, 4)
# 		self.block3 = resnet_block(128, 256, 6)
# 		self.block4 = resnet_block(256, 512, 3)
#
# 	def forward(self, x):
# 		features = []
# 		x = self.conv1(x)
# 		features.append(x)  # 1/4下采样
#
# 		x = self.block1(x)
# 		features.append(x)  # 1/4
#
# 		x = self.block2(x)
# 		features.append(x)  # 1/8
#
# 		x = self.block3(x)
# 		features.append(x)  # 1/16
#
# 		x = self.block4(x)
# 		features.append(x)  # 1/32
#
# 		return features  # 返回5个不同尺度的特征图
#
# def resnet34(num_classes, in_channels=3):
# 	# 也可以使用语句 net = nn.Sequential(b1) 来代替下方
# 	net = nn.Sequential(
# 		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
# 		nn.BatchNorm2d(64),
# 		nn.ReLU(inplace=True),
# 		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 	)
# 	net.add_module("resnet_block1", resnet_block(64, 64, 3, first_block=True))
# 	net.add_module("resnet_block2", resnet_block(64, 128, 4))
# 	net.add_module("resnet_block3", resnet_block(128, 256, 6))
# 	net.add_module("resnet_block4", resnet_block(256, 512, 3))
# 	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
# 	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
# 	# Optionally, initialize weights here
# 	return net
# models/backbone.py

import torch
import torch.nn as nn
# 导入 torchvision 的 resnet34 模型和预训练权重
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class ResNet34FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载 torchvision 的 ResNet-34
        # 使用 weights 参数加载预训练权重，这里使用最新的推荐方式
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet34 = models.resnet34(weights=weights)

        # 移除 ResNet 最后的分类层
        # 我们只需要特征提取部分
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool

        self.layer1 = resnet34.layer1 # ResNet 第一个 block
        self.layer2 = resnet34.layer2 # ResNet 第二个 block
        self.layer3 = resnet34.layer3 # ResNet 第三个 block
        self.layer4 = resnet34.layer4 # ResNet 第四个 block

        # 根据你的需要，可能需要 Freeze 部分层（例如只训练后面的层）
        # 例如，冻结 conv1 和 layer1 的参数
        if pretrained:
             # print("Freezing initial layers of ResNet-34")
             for param in self.conv1.parameters():
                  param.requires_grad = False
             for param in self.bn1.parameters():
                  param.requires_grad = False
             for param in self.layer1.parameters():
                  param.requires_grad = False
             # 根据你的实验需要决定冻结多少层

    def forward(self, x):
        # 按照 ResNet 的前向传播顺序获取中间层特征
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x) # 初始特征，通常不用于检测头

        x = self.layer1(x)
        features.append(x) # layer1 输出，1/4 下采样

        x = self.layer2(x)
        features.append(x) # layer2 输出，1/8 下采样

        x = self.layer3(x)
        features.append(x) # layer3 输出，1/16 下采样

        x = self.layer4(x)
        features.append(x) # layer4 输出，1/32 下采样


        # 如果你的模型输入是 1024x1024，那么这些特征图的尺寸大致是：256x256, 256x256, 128x128, 64x64, 32x32。
        return features

# 你原来的 resnet_block 和 Residual_primary 类如果不再被其他地方使用，可以移除或注释掉。