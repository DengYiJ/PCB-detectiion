import torch
import torch.nn as nn
from param import Embeding_dim
#from resnet34 import ResidualBasicBlockShortcut
from torch import autocast
import torch.nn.functional as F
class PatchEmbedding1(nn.Module):
    def __init__(self,  embed_dim=768, norm_layer=None):
        super(PatchEmbedding1, self).__init__()
        self.patch_size=None  #动态调整patchsize
        self.grid_size = None  # 初始化为 None 或其他默认值
        self.num_patches = 0  # 初始值为 0 或其他默认值
        #动态设置
        self.in_channels = None
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.proj = None
        self.norm = None
     #   self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
      #  self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    # @autocast('cuda')#@autocast() 装饰器被应用于 forward 方法，这意味着在前向传播过程中，PyTorch 会自动选择合适的精度进行计算。
    def forward(self, x):
        # 打印输入的类型（dtype），检查是否为 FP16
        # print(f"Input tensor dtype: {x.dtype}")# FP32
       # x=x.type(torch.float16)
       #  print(f"PaE input tensor dtype: {x.dtype}")
        B, C, H, W = x.shape
        #print(f"PatchEmbedding input shape: {x.shape}")
        # 动态调整 patch_size，确保与输入张量尺寸兼容
        # 动态计算 patch_size
        if self.patch_size is None or (self.patch_size[0] > H or self.patch_size[1] > W):
            self.patch_size = (H // 32, W // 32)
            if self.patch_size[0] < 1:
                self.patch_size = (1, self.patch_size[1])
            if self.patch_size[1] < 1:
                self.patch_size = (self.patch_size[0], 1)

        if self.in_channels !=C:
            self.in_channels = C  # 根据输入张量的通道数动态设置 in_channels
        #print(f"Patch in_channels C is :{self.in_channels}")

        if self.proj is None or self.proj.in_channels != self.in_channels or self.proj.kernel_size != self.patch_size:
            self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            # self.proj = self.proj.to(x.device).half()  # 确保权重在同一个设备上
            self.proj = self.proj.to(x.device)
            # 动态创建归一化层并移动到输入所在的设备

        if self.norm is None:
            if self.norm_layer is not None:
                self.norm = self.norm_layer(self.embed_dim)
            else:
                self.norm = nn.Identity()
            # self.norm = self.norm.to(x.device).half()  # 确保权重在同一个设备上
            self.norm = self.norm.to(x.device)

        patch_height, patch_width = self.patch_size
        self.grid_size = (H // patch_height, W // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 确保输入和权重在同一个设备上
      #  x = x.to(self.proj.weight.device)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        #print(f"PatchEmbedding output shape: {x.shape}")
        # 打印输入的类型（dtype），检查是否为 FP16
        # print(f"PaE output tensor dtype: {x.dtype}")
        # print(f"Output device: {x.device}")
        return x


class PatchEmbedding2(nn.Module):
    def __init__(self, embed_dim=768, norm_layer=None, window_size=256, stride=128):
        super(PatchEmbedding2, self).__init__()
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.window_size = window_size  # 滑动窗口大小
        self.stride = stride  # 滑动步长

        # 初始化动态参数
        self.patch_size = None
        self.grid_size = None
        self.num_patches = 0
        self.in_channels = None
        self.proj = None
        self.norm = None

    def _extract_patches(self, x):
        """提取图像块"""
        B, C, H, W = x.shape
        patches = []

        # 计算需要的padding
        pad_h = (self.window_size - H % self.stride) % self.window_size
        pad_w = (self.window_size - W % self.stride) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 提取patches
        for i in range(0, H + pad_h - self.window_size + 1, self.stride):
            for j in range(0, W + pad_w - self.window_size + 1, self.stride):
                patch = x[:, :, i:i + self.window_size, j:j + self.window_size]
                patches.append(patch)

        # 将所有patches拼接成一个batch
        patches = torch.cat(patches, dim=0)  # [N*B, C, window_size, window_size]
        return patches, (H, W, pad_h, pad_w)

    def _merge_patches(self, patches, original_size, count_map):
        """合并patches回原始尺寸"""
        B, embed_dim, num_patches = patches.shape
        H, W, pad_h, pad_w = original_size

        # 创建输出张量
        merged = torch.zeros((B, embed_dim, H + pad_h, W + pad_w), device=patches.device)

        # 还原每个patch的位置
        idx = 0
        for i in range(0, H + pad_h - self.window_size + 1, self.stride):
            for j in range(0, W + pad_w - self.window_size + 1, self.stride):
                merged[:, :, i:i + self.window_size, j:j + self.window_size] += \
                    patches[:, :, idx:idx + self.window_size * self.window_size].view(
                        B, embed_dim, self.window_size, self.window_size)
                count_map[i:i + self.window_size, j:j + self.window_size] += 1
                idx += self.window_size * self.window_size

        # 平均重叠区域
        merged = merged / count_map.unsqueeze(0).unsqueeze(0)

        # 移除padding
        if pad_h > 0 or pad_w > 0:
            merged = merged[:, :, :H, :W]

        return merged

    # @autocast('cuda')
    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 提取patches
        patches, original_size = self._extract_patches(x)
        num_patches = patches.shape[0] // B

        # 2. 动态设置patch_size参数
        if self.patch_size is None or (self.patch_size[0] > self.window_size or self.patch_size[1] > self.window_size):
            self.patch_size = (self.window_size // 32, self.window_size // 32)
            self.patch_size = (max(1, self.patch_size[0]), max(1, self.patch_size[1]))

        # 3. 更新输入通道数和投影层
        if self.in_channels != C or self.proj is None:
            self.in_channels = C
            self.proj = nn.Conv2d(
                self.in_channels,
                self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ).to(x.device)

        # 4. 更新归一化层
        if self.norm is None:
            self.norm = (self.norm_layer(self.embed_dim) if self.norm_layer else nn.Identity()).to(x.device)

        # 5. 处理每个patch
        patch_height, patch_width = self.patch_size
        self.grid_size = (self.window_size // patch_height, self.window_size // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 6. 投影和变换
        patches = self.proj(patches)  # [N*B, embed_dim, grid_h, grid_w]
        patches = patches.flatten(2).transpose(1, 2)  # [N*B, num_patches, embed_dim]
        patches = self.norm(patches)

        # 7. 重塑为原始batch大小
        patches = patches.view(B, num_patches, self.num_patches, -1)  # [B, N, num_patches, embed_dim]
        patches = patches.view(B, -1, self.embed_dim)  # [B, N*num_patches, embed_dim]

        return patches, original_size

    def reset_parameters(self):
        """重置模型参数"""
        if self.proj is not None:
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 0)
def test_PEb1():
    # 设置超参数
    # 示例使用
    # 实例化 PatchEmbedding 模块
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    patch_embed = PatchEmbedding1(embed_dim=Embeding_dim) # 不指定 in_channels 和 patch_size
    patch_embed=patch_embed
    patch_embedcuda=patch_embed.to(device)
    # 输入张量
    x = torch.randn(1, 3, 224, 224).to(device) #成功移到GPU上，输入FP32，启动混合精度输出变成FP16
    # print(f"x  tensor dtype: {x.dtype}")
    x = patch_embedcuda(x)  # 输出形状为 [8, 1024, 196]

    for name, param in  patch_embedcuda.named_parameters():  # 打印ff模块的模型参数
        print(f"Parameter '{name}' dtype: {param.dtype}")

    # # 再次调用时，输入张量的形状可以不同
    # x = torch.randn(8, 1, 191, 768)
    # x = patch_embed(x)  # 输出形状为 [8, 4, 768]
    #
    # x = torch.randn(8, 1, 5, 768)
    # x = patch_embed(x)  # 输出形状为 [8, ..., ...]
# 调用测试函数
if __name__ == "__main__":
    test_PEb1()