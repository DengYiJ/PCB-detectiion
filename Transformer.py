import math

import torch.nn.functional as F
import torch.nn as nn
#from torch.nn import GELU
import numpy as np
import torch
from functorch.einops import rearrange
from torch import autocast


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    #特征向量长度，多头注意力头数，QKV偏置，注意力dropout比例，输出dropout比例
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        print(f"dim: {dim}, num_heads: {num_heads},scale: {self.scale}")
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)# 一个线性层，用于生成查询（Query）、键（Key）和值（Value）。输入维度为dim，输出维度为dim * 3（因为同时生成Q、K、V）
        self.attn_drop = nn.Dropout(attn_drop)#对注意力权重进行dropout
        self.proj = nn.Linear(dim, dim)#线性层，将注意力机制的输出投影回原始维度dim
        self.proj_drop = nn.Dropout(proj_drop)#对投影输出进行dropout

    def forward(self, x):
        B, N, C = x.shape  #批量大小，序列长度（图像中的patch数量），输入特征向量长度
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#通过线形层生成qkv，将他reshape，然后permute调整维度顺序变成(3, B, num_heads, N, C // num_heads)。
        q, k, v = qkv[0], qkv[1], qkv[2]#提取qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale#计算Q和K的点积，得到注意力权重矩阵。形状为 (B, num_heads, N, N)。 self.scale: 对注意力权重进行缩放，防止梯度消失。
        attn = attn.softmax(dim=-1) #对注意力权重进行softmax操作，使其和为1。形状为 (B, num_heads, N, N)。
        attn = self.attn_drop(attn)#对注意力权重进行Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#: 使用注意力权重对值（V）进行加权求和。形状为 (B, num_heads, N, C // num_heads)。将输出 reshape 为 (B, N, C)
        x = self.proj(x)#通过线性层将输出投影回原始维度C。
        x = self.proj_drop(x) #对投影后的输出进行Dropout
        return x  #返回最终的输出特征x，形状为 (B, N, C)


def testTrans():
# 输入特征 (Batch size=2, Sequence length=10, Feature dimension=64)两批，序列长为10，特征维度为64
    x = torch.randn(4, 163, 768)
    dim=768
# 创建注意力模块 (dim=64, num_heads=8)
    attn = Attention(dim, num_heads=4)

# 前向传播
    output = attn(x)
    print(output.shape)  # 输出: torch.randn(4, 163, 768)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks机器语言程序
    """
    #输入特征维度，隐藏层维度，输出特征维度，激活函数，dropout比例
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  #若未指定，则为in_features
        hidden_features = hidden_features or in_features#同上
        drop_probs = (drop, drop)#dropout比例，应用于两个层

        self.fc1 = nn.Linear(in_features, hidden_features)#第一个全连接层，将输入特征映射到隐藏特征层
        self.act = act_layer()#激活函数
        self.drop1 = nn.Dropout(drop_probs[0])#dropout第一次
        self.fc2 = nn.Linear(hidden_features, out_features)#第二个全连接层，将隐藏层特征映射到输出层
        self.drop2 = nn.Dropout(drop_probs[1])#第二次dropout

    def forward(self, x):
        x = self.fc1(x)#输入->隐藏
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    #输入特征维度，头数，隐藏层维度与输入维度的比值，qkv偏置，MLPdropout比例，注意力机制的dropout比例，droppath的drop比例，激活函数，归一化层
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)#归一化输入特征
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)#注意力机制层
        self.norm2 = norm_layer(dim)#归一化注意力机制输出
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)#MLP层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()#droppath层，用于随即丢弃路径

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))#先归一化，然后注意力机制，然后丢弃path，然后残差链接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=64, block_size=32,reduction_ratio=4):
        super().__init__()
        self.heads = num_heads
        self.block_size = block_size
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Linear(dim, dim)  # 添加投影层
        self.downsample = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1) if dim > 256 else None
        # # 将权重和偏置转换为 FP16
        # self.qkv.weight = nn.Parameter(self.qkv.weight.to(torch.float16))
        # self.qkv.bias = nn.Parameter(self.qkv.bias.to(torch.float16))
        # self.proj.weight = nn.Parameter(self.proj.weight.to(torch.float16))
        # self.proj.bias = nn.Parameter(self.proj.bias.to(torch.float16))
    # @torch.amp.autocast(device_type='cuda')
    def forward(self, x):

        # if self.downsample and x.shape[1] > 4096:
        #     x = x.transpose(1, 2)
        #     x = self.downsample(x)
        #     x = x.transpose(1, 2)

        B, N, C = x.shape
        # print(f"x shape: {x.shape}, dtype: {x.dtype}")
        qkv = self.qkv(x)
        # print(f"qkv shape: {qkv.shape}, dtype: {qkv.dtype}")  # 调试 qkv 形状和类型

        # 将 qkv 分头
        qkv = rearrange(qkv, 'b n (h d) -> b h n d', h=self.heads)
        q, k, v = qkv.chunk(3, dim=-1)

        # 合并头以进行分块
        q = rearrange(q, 'b h n d -> b n (h d)')
        k = rearrange(k, 'b h n d -> b n (h d)')
        v = rearrange(v, 'b h n d -> b n (h d)')

        output = self._sparse_block_attention(q, k, v, N)

        # 转换回多头格式并投影
        output = rearrange(output, 'b n (h d) -> b n h d', h=self.heads)
        output = rearrange(output, 'b n h d -> b n (h d)')
        # print(f"rearranged output shape: {output.shape}, dtype: {output.dtype}")  # 调试重排后的输出
        output = self.proj(output)

        # print("Output dtype:", output.dtype)#Output dtype: torch.float16
        return output

    def _dynamic_blocking(self, x, seq_len):
        """集成动态分块逻辑"""
        block_size = self.block_size
        num_blocks = seq_len // block_size
        remainder = seq_len % block_size

        # 尾部填充处理
        if remainder > 0:
            pad_size = block_size - remainder
            x = F.pad(x, (0, 0, 0, pad_size))  # 填补长度到 L+pad
            num_blocks += 1
        else:
            pad_size = 0

        # 分块重塑 (B, num_blocks, block_size, C)
        x_blocks = x.view(-1, num_blocks, block_size, x.size(-1))
        return x_blocks, num_blocks, pad_size

    def _sparse_block_attention(self, q, k, v, seq_len):
        """改进后的分块注意力"""
        B, N, C = q.shape

        # Step 1: 动态分块处理
        q_blocks, num_blocks, pad_size = self._dynamic_blocking(q, seq_len)
        k_blocks, _, _ = self._dynamic_blocking(k, seq_len)
        v_blocks, _, _ = self._dynamic_blocking(v, seq_len)

        # Step 2: 滑动窗口处理
        outputs = []
        for i in range(num_blocks):
            # 计算当前块的邻域范围（按块数计算）
            start = max(0, i - self.window_size)
            end = min(num_blocks, i + self.window_size + 1)

            # 提取当前查询块和邻域键值块
            q_block = q_blocks[:, i]  # (B, block_size, C)
            k_blocks_window = k_blocks[:, start:end]  # (B, window_blocks, block_size, C)
            v_blocks_window = v_blocks[:, start:end]  # (B, window_blocks, block_size, C)

            # 计算块间注意力
            attn = torch.einsum('bsc,bwsc->bsw', q_block, k_blocks_window) * self.scale
            attn = F.softmax(attn, dim=-1)

            # 加权聚合
            out = torch.einsum('bsw,bwsc->bsc', attn, v_blocks_window)
            outputs.append(out)

        # Step 3: 合并输出并裁剪填充
        output = torch.cat(outputs, dim=1)  # (B, num_blocks*block_size, C)
        if pad_size > 0:
            output = output[:, :-pad_size]  # 裁剪填充部分
        # print(f"output after clipping: {output.shape}")  # 调试信息
        return output

    class Block(nn.Module):
        """优化后的Block保持原有接口"""

        def __init__(self, dim, num_heads, mlp_ratio=4.,
                     window_size=64, block_size=64,
                     drop=0., attn_drop=0., drop_path=0.1):
            super().__init__()
            #第一层，标准化+稀疏注意力
            self.norm1 = nn.LayerNorm(dim)
            self.attn = SparseAttention(
                dim,
                num_heads=num_heads,
                window_size=window_size,
                block_size=block_size
            )
            #第二层：标准化+MLP
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
            #残差连接控制
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x):
            #第一层残差
            x = x + self.drop_path(self.attn(self.norm1(x)))
            #第二层残差
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

def test_sparse_attention():
    # 定义测试参数
    dim = 196
    num_heads = 4
    window_size = 64
    block_size = 32

    # 构造输入张量
    batch_size = 1
    seq_len = 1025
    x = torch.randn(batch_size, seq_len, dim).half().cuda()
    # print(f"x input tensor dtype: {x.dtype}") #FP32
    # 初始化 SparseAttention 模块
    attention = SparseAttention(dim, num_heads, window_size, block_size).half().cuda()#取半了

    # 执行前向传播
   # with torch.autocast(device_type="cuda"):
    output = attention(x)  #Output dtype: torch.float16
    print(f"sparse output shape: {output.shape}, dtype: {output.dtype}") #([1,1025,196])(B,N,C)
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, dim), "输出形状不匹配"
    for name, param in  attention.named_parameters():  # 打印ff模块的模型参数
        print(f"Parameter '{name}' dtype: {param.dtype}")

if __name__ == "__main__":
    test_sparse_attention()