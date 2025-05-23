import torch.nn.functional as F
import torch
from torch import nn, autocast


# conv = SKConv(64, 32, 3, 8, 2)   output:`[batch_size, features, H, W]`。
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: 输入通道维度
            WH: 输入特征图的空间维度
            M: 分支的数量
            G: 卷积组的数量
            r: 计算d，向量s的压缩倍数，C/r
            stride: 步长，默认为1
            L: 矢量z的最小维度，默认为32
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        # 使用不同kernel size的卷积，增加不同的感受野
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
               # nn.ReLU(inplace=False),
                nn.GELU()
            ))
        # 全局平均池化
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        # 全连接层
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
    # @autocast('cuda')
    def forward(self, x):
        ''' Split操作'''
        x = x.to(torch.float32)
        # print(f"[SKConv] Input shape: {x.shape}, dtype={x.dtype}")  # 输入验证[2](@ref)
        for i, conv in enumerate(self.convs):
          # fea = conv(x).unsqueeze_(dim=1)
            # 修改后代码（避免原地操作）：
            fea = conv(x).unsqueeze(dim=1)  # 创建新张量
            # print(f"[SKConv] Branch {i} output range: [{fea.min():.3f}, {fea.max():.3f}]")  # 卷积分支数值监测[1]
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        ''' Fuse操作'''
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        # print(f"[SKConv] After GAP: shape={fea_s.shape}")  # 确认降维正确性[2]
        fea_z = self.fc(fea_s)
        assert not torch.isnan(fea_z).any(), "NaN in fea_z!"
        # print(f"[SKConv] fea_z range: [{fea_z.min():.3f}, {fea_z.max():.3f}]")  # 监测全连接层输出[1]
        ''' Select操作'''
        for i, fc in enumerate(self.fcs):
            # fc-->d*c维
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        # 计算attention权重
        # attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.clamp(min=-10, max=10)
        # attention_vectors = attention_vectors - attention_vectors.max(dim=1, keepdim=True).values
        attention_vectors = self.softmax(attention_vectors)
        # print(f"[SKConv] Attention weights sum: {attention_vectors.sum(dim=1)}")  # 验证概率分布[
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # 最后一步，各特征图与对应的注意力权重相乘，得到输出特征图V
        fea_v = (feas * attention_vectors).sum(dim=1)
        # print(f"[SKConv] Output shape: {fea_v.shape},dtype={fea_v.dtype}, has NaN: {torch.isnan(fea_v).any()}")  # 最终输出验证[5]
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: 输入通道维度
            out_features: 输出通道维度
            WH: 输入特征图的空间维度
            M: 分支的数量
            G: 卷积组的数量
            r: 计算d，论文中向量s的压缩倍数，C/r
            mid_features: 步长不为1的中间卷积的通道维度，默认为out_features/2
            stride: 步长，默认为1
            L: 论文中矢量z的最小维度，默认为32
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            # SKConv替代了ResNext中3*3卷积部分
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKNet(nn.Module):
    def __init__(self, class_num):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )  # 32x32
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        )  # 32x32
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU()
        )  # 16x16
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU()
        )  # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


# 测试用例
def test_skconv():
    # 设置超参数
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    WH = 8
    M = 2
    G = 2
    r = 2
    stride = 1
    L = 32

    # 输入张量
    x = torch.randn(batch_size, channels, height, width).cuda() #输入FP32

    # 初始化 SKConv 模块
    skconv = SKConv(features=channels, WH=WH, M=M, G=G, r=r, stride=stride, L=L).cuda()

    # 前向传播
    output = skconv(x)

    # 验证输出形状
    expected_output_shape = (batch_size, channels, height, width)
    assert output.shape == expected_output_shape, f"输出形状不匹配，期望 {expected_output_shape}，实际 {output.shape}"

    for name, param in skconv.named_parameters():  # 打印ff模块的模型参数
        print(f"Parameter '{name}' dtype: {param.dtype}")

    print("SKConv 测试通过！")

if __name__ == "__main__":
    test_skconv()