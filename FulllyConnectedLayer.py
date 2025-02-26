import torch
import torch.nn as nn
import torch.nn.functional as F
class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc=nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x=self.fc(x)
        x=F.relu(x)
        return x

    '''先搭建模型，后forward动态'''
    # 示例
if __name__ == "__main__":
        # 输入特征维度和输出特征维度
        input_features = 1024
        output_features = 512

        # 创建全连接层
        fc_layer = FullyConnected(input_features, output_features)

        # 输入数据
        x = torch.randn(32, 1024)  # 32个样本，每个样本1024个特征

        # 前向传播
        output = fc_layer(x)
        print(output.shape)  # 输出: torch.Size([32, 512])