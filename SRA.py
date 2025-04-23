# -*- coding: utf-8 -*-
import torch
from torch import nn


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, D = x.shape  # N=h*w
        q = self.q(x).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, D, H, W)
            x_ = self.sr(x_).reshape(B, D, -1).permute(0, 2, 1)  # 这里x_.shape = (B, N/R^2, D)
            x_ = self.norm(x_)
            # 因为做检测分割的图片的分辨率都很大, N也就很大
            # 这样也是为了不再需要K@V，因为计算量较大
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)

        return x


# class Attention(nn.Module):
#     fused_attn: torch.jit.Final[bool]
#
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             sr_ratio=1,
#             linear_attn=False,
#             qkv_bias=True,
#             attn_drop=0.,
#             proj_drop=0.
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim  # 64
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = use_fused_attn()
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         if not linear_attn:
#             self.pool = None
#             if sr_ratio > 1:
#                 self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # 64,64,8,8
#                 self.norm = nn.LayerNorm(dim)
#             else:
#                 self.sr = None
#                 self.norm = None
#             self.act = None
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(7)
#             self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#             self.norm = nn.LayerNorm(dim)
#             self.act = nn.GELU()
#
#     def forward(self, x, feat_size: List[int]):
#         B, N, C = x.shape  # (1,3136,64)
#         H, W = feat_size  # (56,56)
#         q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (1,3136,64)->(1,3136,1,64)->(1,1,3136,64)
#
#         if self.pool is not None:
#             x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (1,64,3136)->(1,64,56,56)
#             x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)  # (1,64,7,7)->(1,64,49)->(1,49,64)
#             x = self.norm(x)
#             x = self.act(x)
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         else:
#             if self.sr is not None:
#                 x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (1,3136,64)->(1,64,3136)->(1,64,56,56)
#                 x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)  # (1,64,7,7)->(1,64,49)->(1,49,64)
#                 x = self.norm(x)
#                 kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1,
#                                                                                          4)  # (1,49,128)->(1,49,2,1,64)->(2,1,1,49,64)
#             else:
#                 kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv.unbind(0)  # (1,1,49,64),(1,1,49,64)
#
#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)  # (1,1,3136,64) @ (1,1,64,49) -> (1,1,3136,49)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v  # (1,1,3136,49) @ (1,1,49,64) -> (1,1,3136,64)
#
#         x = x.transpose(1, 2).reshape(B, N, C)  # (1,3136,1,64)->(1,3136,64)
#         x = self.proj(x)  # (1,3136,64)
#         x = self.proj_drop(x)
#         return x

# x = torch.rand(4, 224 * 128, 256)
# attn = SpatialReductionAttention(dim=256, sr_ratio=2)
# output = attn(x, H=224, W=128)
# print(f"output.shape:{output.shape}")   #[4,28672,128]