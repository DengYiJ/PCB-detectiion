import torch
import torch.nn as nn
import torch.nn.functional as F

class pyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pyramid, self).__init__()
        self.channel_unify = nn.Conv2d(in_channels, 64, kernel_size=1)  #统一尺寸
        self.se_block=se_block(64)#添加se模块
        self.smooth_cov = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=False))
                                                # 添加下采样卷积层
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )#下采样层
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, 0)

    def tensor_add(self,x,y):
        #将X插值到y相同的空间尺寸
        #用于上采样 8x8->16x16
        x_resized = F.interpolate(x, size=(y.size(2),y.size(3)), mode='bilinear')
        # attention_weight = self.attention[0](y)  # 生成注意力权重
        # return y * attention_weight + x_resized * (1 - attention_weight)
        return y+x_resized

    def forward(self, features):
        output=[]
        # print("Pyrami Input features shapes:")
        # for feature in features:
        #     print(feature.shape)

        # for feature in features:
        #     x=self.sequential(feature)
        #     x=self.relu(x)
        #     output.append(x)
        for i, feature in enumerate(features):
            x = self.channel_unify(feature)
           # print(f"unified output {i + 1}: {x.shape}")
            #x = self.relu(x)
            # print(f"ReLU output {i + 1}: {x.shape}")
            output.append(x)
        # print("channel unify features shapes:")
        # for feature in output:
        #     print(feature.shape)
        #自上而下,横向链接
        # p5=output[4]
        # p4=self._umsample_add(p5,output[3])
        # p3=self._umsample_add(p4,output[2])
        # p2=self._umsample_add(p3,output[1])
        # p1=self._umsample_add(p2,output[0])
        p5=output[4]
        # print(f"p5 shape {p5.shape}")
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
        # print(f"p1 smooth shape {p1.shape}")
        # print(f"p2 smooth shape {p2.shape}")
        # print(f"output1 smooth shape {output[1].shape}")
        #print("平滑处理成功")
        #out_channels = 6
        # MultiScaleEmbedding 操作
        b1 = p1
        # print(f"b1  shape {b1.shape}")
        # print(f"p2  shape {p2.shape}")
        # print(f"output1  shape {output[1].shape}")
        b2 = self.se_block(p2 + b1 + output[1])   # 三个特征图相加后通过SE
        # print(f"b2:{b2.shape}")
        b3 = self.se_block(p3 + self.downsample(b2) + output[2])
        b4 = self.se_block(p4 + self.downsample(b3) + output[3])
        b5 = self.tensor_add(self.downsample(b4), p5)  # 在 b5 处进行相加 操作
        b6 = self.maxpool(b5)  #宽高除以2
        # print("ME succeed")
        feature_list=[b1,b2,b3,b4,b5,b6]
        # for i, feature in enumerate(feature_list):
        #     print(f"b{i + 1} shape: {feature.shape}")

        return  feature_list #返回一个带有5个(B,D,H,W)的list，输出的D为经过me卷积，等于me的outchannels6

 
#SENet,通道注意力机制
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def test_pyramid_module():
    # 测试参数设置
    batch_size = 2
    in_channels = 64
    out_channels = 32
    height = 64
    width = 64

    # 生成5个不同分辨率的输入特征图
    input_features = []
    for scale in range(5):
        h = height // (2 ** scale)
        w = width // (2 ** scale)
        feature = torch.randn(batch_size, in_channels, h, w)
        input_features.append(feature)

    # 打印输入特征图信息
    # print("Input features:")
    # for i, feat in enumerate(input_features):
    #     print(f"Feature {i + 1}: shape={feat.shape}")

    # 初始化金字塔模块
    pyramid_model = pyramid(in_channels, out_channels)

    # 前向传播
    output_features = pyramid_model(input_features)

    # 验证输出
    print("\nOutput features:")
    for i, out in enumerate(output_features):
        print(f"Output {i + 1}: shape={out.shape}")
        # 检查输出形状
        assert out.shape[0] == batch_size, f"Batch size mismatch in output {i + 1}"
        assert out.shape[1] == out_channels, f"Channel mismatch in output {i + 1}"

    # 检查输出数量是否正确
    assert len(output_features) == 6, "Output feature count mismatch"

    print("\nPyramid module test passed!")


if __name__ == "__main__":
    test_pyramid_module()