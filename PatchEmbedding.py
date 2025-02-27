import torch
import torch.nn as nn
#from resnet34 import ResidualBasicBlockShortcut
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224,224),patch_size=16,in_channels=3,embed_dim=768,norm_layer=None):
        super(PatchEmbedding, self).__init__()
        img_height, img_width = img_size
        patch_height, patch_width = patch_size,patch_size
        self.img_size=img_size
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = (patch_height, patch_width)
        self.grid_size = (self.img_height//patch_height, self.img_width//patch_width)#计算每个维度上的 patch 数量
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(embed_dim)if norm_layer else nn.Identity()

    def forward(self, x):
        B,C,H,W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x)
        return x

# 示例使用
img_height = 1586
img_width = 3034
patch_size = 16
in_chans = 3
embed_dim = 768
num_classes = 10

# 创建 PatchEmbed 模块
patch_embed = PatchEmbedding(img_size=(img_height, img_width), patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim)

# 创建一个示例输入张量
x = torch.randn(1, in_chans, img_height, img_width)

# 前向传播
x = patch_embed(x)  # [B, N, D]#(batch_size, num_patches, embed_dim)

print(x.shape)  # 输出形状