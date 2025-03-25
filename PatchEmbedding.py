import torch
import torch.nn as nn
from param import Embeding_dim
#from resnet34 import ResidualBasicBlockShortcut
from torch import autocast
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224,224),patch_size=16,in_channels=3,embed_dim=768,norm_layer=None):
        super(PatchEmbedding, self).__init__()
        img_height, img_width = img_size
        patch_height, patch_width = patch_size,patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            self.patch_size = patch_size
        else:
            raise ValueError("patch_size must be an integer or a tuple of two integers")
        #self.img_size=img_size
        #self.img_height = img_height
        #self.img_width = img_width
        #self.patch_size = (patch_height, patch_width)
        #self.grid_size = (self.img_height//patch_height, self.img_width//patch_width)#计算每个维度上的 patch 数量
        #self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(embed_dim)if norm_layer else nn.Identity()

    def forward(self, x):
        #print(f"PatchEmbedding input shape: {x.shape}")
        B,C,H,W = x.shape
       # assert H == self.img_size[0] and W == self.img_size[1],f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        patch_height, patch_width = self.patch_size
        self.img_size = (H, W)
        self.grid_size = (H // patch_height, W // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        x = self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x)
        #print(f"PatchEmbedding output shape: {x.shape}")
        return x

    def test_PEb():
    # 设置超参数
    # 示例使用
        img_height = 1586  # 1600 reshape后
        img_width = 3034  # 3040
        patch_size = 160  # 16→160
        in_chans = 3  # RGB三通道
        embed_dim = 768
    # dataset return 了images，他的张量是（[4，3，224，224]）【4，3，1586，3034】

    # 创建 PatchEmbed 模块
        patch_embed = PatchEmbedding(img_size=(img_height, img_width), patch_size=patch_size, in_channels=in_chans,
                                 embed_dim=embed_dim)

    # 创建一个示例输入张量
        x = torch.randn(4, in_chans, img_height, img_width)

    # 前向传播
        x = patch_embed(x)  # [B, N, D]#(batch_size, num_patches, embed_dim)

        print(x.shape)  # 输出形状，torch.Size([4, 18711, 768])→([4, 162, 768])

#测试代码
    #test_PEb()

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
    @autocast('cuda')#@autocast() 装饰器被应用于 forward 方法，这意味着在前向传播过程中，PyTorch 会自动选择合适的精度进行计算。
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