import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalPropagation(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0),
                              groups=dim)
        # 改为1D深度可分离卷积
        self.conv1d = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim  # 深度可分离卷积
        )
        # 添加通道混合
        self.channel_mixer = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)  # [B, C, N]

        # 1. 转换为[B, C, N, 1]的4D张量
# with torch.amp.autocast(device_type="cuda"):
            # x = x.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]
        # 2. 使用1D卷积处理(通过设置kernel_size=(k,1))
        x =self.conv1d(x)
        # 3. 移除虚拟维度并归一化
        #     x = x.squeeze(-1).transpose(1, 2)  # [B, N, C]
        x = x.transpose(1, 2)  # [B, N, C]
        x = self.channel_mixer(x)
        # x = self.norm(x)
        return self.norm(x)