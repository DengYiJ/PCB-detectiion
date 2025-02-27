import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet基本的Bottleneck类
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩增倍数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # shortcut
        out = self.relu(out)
        return out


class FPN(nn.Module):
    '''
    FPN需要初始化一个list，代表ResNet每一个阶段的Bottleneck的数量
    '''

    def __init__(self, layers):
        super(FPN, self).__init__()
        # 构建C1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # 自下而上搭建C2、C3、C4、C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5减少通道，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)

        # 3*3卷积融合
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)

        # 横向连接，保证每一层通道数一致
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

    # 构建C2到C5
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        # 如果步长不为1，进行下采样
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        # 更新输入输出层
        self.inplanes = planes * Bottleneck.expansion
        # 根据block数量添加bottleneck的数量
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

        # 自上而下上采样

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        # 逐个元素相加
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 自上而下，横向连接
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # 卷积融合，平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5

class pyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pyramid, self).__init__()
        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(5)])#为con_list添加5个conv2d

        self.sequential = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),)
        self.relu = nn.ReLU(inplace=True)
        self.me = MultiScaleEmbedding(in_channels, out_channels)  # 添加 MultiScaleEmbedding 模块
        self.maxpool = nn.MaxPool2d(2, 2, 0)
    def _umsample_add(self,x,y):
        _, _, H, W = y.shape
        #逐个元素相加
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, features):
        output=[]
        for i,feature in enumerate(features):
            x=self.conv_list[i](feature)
            x=self.relu(x)
            output.append(x)
        #自上而下,横向链接
        p5=output[5]
        p4=self._umsample_add(p5,output[4])
        p3=self._umsample_add(p4,output[3])
        p2=self._umsample_add(p3,output[2])
        p1=self._umsample_add(p2,output[1])

        #卷积融合，平滑处理
        p5=self.sequential(p5)
        p4=self.sequential(p4)
        p3=self.sequential(p3)
        p2=self.sequential(p2)
        p1=self.sequential(p1)

        # MultiScaleEmbedding 操作
        b1 = p1
        b2 = self.me([p2, b1,output[1]])   # 在 b2 处进行 ME 操作并添加残差链接
        b3 = self.me([p3, b2,output[2]])   # 在 b3 处进行 ME 操作并添加残差链接
        b4 = self.me([p4, b3,output[3]])   # 在 b4 处进行 ME 操作并添加残差链接
        b5 = self._umsample_add(p5, b4)  # 在 b5 处进行相加 操作
        b6 = self.maxpool(b5)

        feature_list=[b1,b2,b3,b4,b5,b6]
        for i, feature in enumerate(feature_list):
            print(f"b{i + 1} shape: {feature.shape}")

        return  feature_list

class MultiScaleEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleEmbedding, self).__init__()
        self.weights=nn.Parameter(torch.ones(3))# 假设每次 ME 操作有两个输入
        self.normalize=nn.softmax(dim=0)
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x_list):
        # x_list 是一个列表，包含3个输入特征图
        weights = self.normalize(self.weights)  # 归一化权重
        output = weights[0] * x_list[0] + weights[1] * x_list[1] +weights[2]*x_list[2] # 加权求和
        output = self.conv(output)  # 通过 1x1 卷积调整通道数
        return output
