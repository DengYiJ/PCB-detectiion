import torch
import torch.nn as nn
import torch.nn.functional as F

class pyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pyramid, self).__init__()
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.Sigmoid()
            ) for _ in range(5)
        ])
        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(5)])#为con_list添加5个conv2d

        self.smooth_cov = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=False))
        self.relu = nn.ReLU(inplace=True)
        self.me = MultiScaleEmbedding(out_channels, out_channels)  # 添加 MultiScaleEmbedding 模块
        self.maxpool = nn.MaxPool2d(2, 2, 0)

    def _umsample_add(self,x,y):
        _, _, H, W = y.shape
        #逐个元素相加
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def tensor_add(self,x,y):
        # 将 x 插值到与 y 相同的长度
        # print(f"y shape: {y.size(-1)}")
        # x = F.interpolate(x, size=y.size(-1), mode='linear').squeeze(1)
        # return x+y
        # x_resized = F.interpolate(x, size=y.size(-1), mode='linear').squeeze(1)
        # attention_weight = self.attention[0](y)  # 生成注意力权重
        # return y * attention_weight + x_resized * (1 - attention_weight)
        x_resized = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
        attention_weight = self.attention[0](y)
        return y * attention_weight + x_resized * (1 - attention_weight)

    def forward(self, features):
        output=[]
        # print("Input features shapes:")
        # for feature in features:
        #     print(feature.shape)
        for i, feature in enumerate(features):
            x = self.conv_list[i](feature)
            # print(f"sequential output {i + 1}: {x.shape}")
            #x = self.relu(x)
            # print(f"ReLU output {i + 1}: {x.shape}")
            output.append(x)

        #自上而下,横向链接
        p5=output[4]
        #print(f"p5 shape {p5.shape}")
        p4=self.tensor_add(p5,output[3])# 应该是高层的往下加，也就是要把输出变成低层;左高右低，把左缩到右然后相加
        # print(f"p4 shape {p4.shape}") #成功，说明add没错
        p3=self.tensor_add(p4,output[2])
        p2=self.tensor_add(p3,output[1])
        p1=self.tensor_add(p2,output[0])
        # print(f"p1 shape {p1.shape}")  # 成功，说明add没错

        #卷积融合，平滑处理
        p5=self.smooth_cov(p5)
        p4=self.smooth_cov(p4)
        p3=self.smooth_cov(p3)
        p2=self.smooth_cov(p2)
        p1=self.smooth_cov(p1)
        #print("平滑处理成功")
        #out_channels = 6
        # MultiScaleEmbedding 操作
        b1 = p1
        b2 = self.me([p2, b1,output[1]])   # 在 b2 处进行 ME 操作并添加残差链接 1315+1025+1025
        b3 = self.me([p3, b2,output[2]])   # 在 b3 处进行 ME 操作并添加残差链接
        b4 = self.me([p4, b3,output[3]])   # 在 b4 处进行 ME 操作并添加残差链接
        b5 = self.tensor_add(b4, p5)  # 在 b5 处进行相加 操作
        # b6 = self.maxpool(b5)
        if b5.size(-1) < 2 or b5.size(-2) < 2:  # 如果宽或高小于2
            # 方案1: 直接跳过maxpool
            # b6 = b5
            # 或者方案2: 使用自适应池化
            b6 = F.adaptive_avg_pool2d(b5, (1,1))
        else:
            b6 = self.maxpool(b5)
        # print("ME succeed")
        # feature_list=[b1,b2,b3,b4,b5,b6]
        # for i, feature in enumerate(feature_list):
        #     print(f"b{i + 1} shape: {feature.shape}")

        return  [b1,b2,b3,b4,b5,b6] #返回一个带有5个(B,D,H,W)的list，输出的D为经过me卷积，等于me的outchannels6

class MultiScaleEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))
        self.normalize = nn.Softmax(dim=0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 改为Conv2d

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 1), out_channels, 1),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(out_channels)  # 改为BatchNorm2d
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        # 统一特征尺寸
        target_size = x_list[0].shape[-2:]
        aligned_features = []
        for x in x_list:
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(x)

        # 特征融合
        weights = self.normalize(self.weights)
        output = sum(w * x for w, x in zip(weights, aligned_features))
        output = self.conv(output)

        # 应用通道注意力
        attention = self.channel_attention(output)
        output = output * attention

        output = self.bn(output)
        output = self.relu(output)

        return output

#需要测试金字塔的输入输出，写一个例程，输入是一个list包含5个张量(B,D,H,W)
# 测试代码
if __name__ == "__main__":
    # 设置超参数
    B = 2  # 批处理大小
    D = 3  # 输入通道数（in_channels）
    N= 64 # 特征图的高度和宽度
    in_channels = D
    out_channels = 6  # 输出通道数
    MEin_channels=out_channels
    MEout_channels=out_channels
    # 生成随机输入特征图列表
    features = []
    for _ in range(5):
        feature = torch.randn(B, D, N)  # 输入带有5个(B, D, H, W)的list
        features.append(feature)
    #print(features)
    # 初始化金字塔模块
    pyramid_model = pyramid(in_channels, out_channels)

    # 测试前向传播
    output_features = pyramid_model(features)

    # 检查输出是否正确
    print("测试完成！")
'''output:b1 shape: torch.Size([2, 6, 4, 4])
          b2 shape: torch.Size([2, 6, 4, 4])
          b3 shape: torch.Size([2, 6, 4, 4])
          b4 shape: torch.Size([2, 6, 4, 4])
          b5 shape: torch.Size([2, 6, 4, 4])
          b6 shape: torch.Size([2, 6, 2, 2])'''