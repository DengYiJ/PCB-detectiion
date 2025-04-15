from torch import nn

# 18、34版本残差块
'''
@:param
		input_channels: 输入通道数
		num_channels: 输出通道数
		use_1x1conv: 是否使用 1x1conv (使用就说明是卷积映射残差块, 要变换尺寸)
		strides: 步长
'''
class Residual_primary(nn.Module):
	def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
		# 默认 use_1x1conv=False, strides=1
		super(Residual_primary, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
		self.bn1 = nn.BatchNorm2d(num_channels)  # BN层
		self.relu1 = nn.ReLU(inplace=True)  # Inplace ReLU to save memory
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_channels)
		# 如果用了 1x1conv 就说明是卷积映射残差块, 要变换尺寸
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
			self.bn3 = nn.BatchNorm2d(num_channels)
		else:
			self.conv3 = None

	def forward(self, X):
		Y = self.conv1(X)
		Y = self.bn1(Y)
		Y = self.relu1(Y)
		Y = self.conv2(Y)
		Y = self.bn2(Y)
		if self.conv3:
			X = self.conv3(X)
			X = self.bn3(X)  # If using 1x1 conv, also apply BN
		Y += X
		Y = nn.ReLU(inplace=True)(Y)  # Optionally, you can move this ReLU outside the Residual class
		return Y

# 初始卷积层和池化层
b1 = nn.Sequential(
	nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Removed bias since we have BN, 输入通道为3表示彩色图像
	nn.BatchNorm2d(64),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 大残差结构
'''
@:param
		input_channels: 输入通道数
		num_channels: 输出通道数
		num_residuals: 残差块的个数
		first_block: 是否是第一个大残差结构
@:return
		nn.Sequential(*blk): 大残差结构
'''
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
	blk = []
	for i in range(num_residuals):
		stride = 2 if i == 0 and not first_block else 1  # 从第二个大残差结构开始, 结构中的第一个残差块一般都会尺寸减半, 即 stride=2
		use_1x1conv = i == 0 and not first_block  # use_1x1conv = False/True, 从第二个大残差结构开始, 结构中的第一个残差块都是卷积映射残差块
		if i == 0:
			blk.append(Residual_primary(in_channels, out_channels, use_1x1conv=use_1x1conv, strides=stride))
		else:
			blk.append(Residual_primary(out_channels, out_channels, strides=stride))
	return nn.Sequential(*blk)


# ResNet-18
# def resnet18(num_classes, in_channels=3):
# 	# net = nn.Sequential(b1)
# 	net = nn.Sequential(
# 		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
# 		nn.BatchNorm2d(64),
# 		nn.ReLU(inplace=True),
# 		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 	)
# 	net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
# 	net.add_module("resnet_block2", resnet_block(64, 128, 2))
# 	net.add_module("resnet_block3", resnet_block(128, 256, 2))
# 	net.add_module("resnet_block4", resnet_block(256, 512, 2))
# 	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
# 	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
# 	# Optionally, initialize weights here (e.g., nn.init.kaiming_normal_(net[0].conv1.weight, mode='fan_out', nonlinearity='relu'))
# 	return net


# ResNet-34
class ResNet34_FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 四个残差块阶段
        self.block1 = resnet_block(64, 64, 3, first_block=True)
        self.block2 = resnet_block(64, 128, 4)
        self.block3 = resnet_block(128, 256, 6) 
        self.block4 = resnet_block(256, 512, 3)
        
    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)  # 1/4下采样
        
        x = self.block1(x)
        features.append(x)  # 1/4
        
        x = self.block2(x)
        features.append(x)  # 1/8
        
        x = self.block3(x) 
        features.append(x)  # 1/16
        
        x = self.block4(x)
        features.append(x)  # 1/32
        
        return features  # 返回5个不同尺度的特征图

def resnet34(num_classes, in_channels=3):
	# 也可以使用语句 net = nn.Sequential(b1) 来代替下方
	net = nn.Sequential(
		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
	)
	net.add_module("resnet_block1", resnet_block(64, 64, 3, first_block=True))
	net.add_module("resnet_block2", resnet_block(64, 128, 4))
	net.add_module("resnet_block3", resnet_block(128, 256, 6))
	net.add_module("resnet_block4", resnet_block(256, 512, 3))
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
	# Optionally, initialize weights here
	return net
