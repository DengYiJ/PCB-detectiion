# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class LocalAggregation(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction_ratio=8):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        # 改为1D注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction_ratio, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(dim, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # # 输入x形状: [B, N, C]
        # with torch.amp.autocast(device_type='cuda'):
        x = x.transpose(1, 2)  # [B, C, N]

        for layer in self.local_conv:
            if isinstance(layer, nn.Conv1d):
                layer.weight.data = layer.weight.data.to(x.dtype)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(x.dtype)
        # 1D卷积处理
        local_feat = self.local_conv(x)

        # 1D注意力
        channel_weight = self.channel_attention(local_feat)
        spatial_weight = self.spatial_attention(local_feat * channel_weight)

        # 输出恢复BNC
        x = (local_feat * channel_weight * spatial_weight).transpose(1, 2)
        return x



def test_local_aggregation():
    # 测试参数
    batch_size = 4
    num_patches = 196  # 14x14的patch
    dim = 768
    kernel_size = 3

    # 创建测试输入 (B, N, C)
    x = torch.randn(batch_size, num_patches, dim)

    # 初始化模块
    local_agg = LocalAggregation(dim=dim, kernel_size=kernel_size)

    # 前向传播测试
    output = local_agg(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 验证形状是否匹配
    assert output.shape == x.shape, "输入输出形状不匹配"

    # 测试非方形patch情况
    non_square_patches = 200  # 非完全平方数
    x_ns = torch.randn(batch_size, non_square_patches, dim)
    output_ns = local_agg(x_ns)
    print(f"\n非方形patch输入形状: {x_ns.shape}")
    print(f"非方形patch输出形状: {output_ns.shape}")
    assert output_ns.shape == x_ns.shape, "非方形patch处理失败"

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_local_aggregation()