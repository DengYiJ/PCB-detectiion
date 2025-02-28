import torch.nn.functional as F
import torch.nn as nn
#from torch.nn import GELU
import numpy as np
import torch
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



