import torch
from torch import nn
import torch.nn.functional as F
from param import Droprate

#torch.randn(4, 18711,768)
class PositionEmbeddingDynamic(nn.Module):#inputshape是宽x高，num_features是序列长度18711，num_features是patch_embedding的em_dim；num_features 是指嵌入维度（embed_dim），即 patch_embedding 的输出特征维度。
    def __init__(self,imgH,imgW,num_features,num_patches,patch_size):#patch_size是16x16的滑窗？
        super(PositionEmbeddingDynamic, self).__init__()
        #self.input_size = input_size
        #self.output_size = output_size
        self.imgH = imgH
        self.imgW = imgW
        self.num_features = num_features
        self.num_patches =num_patches  #将图片分成几块？应该来自PositionEmbedding输出的第二个18711
        self.new_feature_shape = [int(imgH// patch_size), int(imgW // patch_size)]
        self.old_feature_shape = [int(imgH // patch_size), int(imgW // patch_size)]

        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        # --------------------------------------------------------------------------------------------------------------------#
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        # --------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为[196, 768]。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        # --------------------------------------------------------------------------------------------------------------------#
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, num_features))
        self.pos_drop = nn.Dropout(p=Droprate)

    def forward(self, x):
    #    x=self.PatchEmbedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token, x), dim=1)

        cls_token_pe=self.pos_embed[:,0:1,:]
        img_token_pe=self.pos_embed[:,1:,:]

        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        x = self.pos_drop(x + pos_embed)
        return x

class PositionEmbeddingStatic(nn.Module):#num_features是序列长度18711，num_features是patch_embedding的em_dim；num_features 是指嵌入维度（embed_dim），即 patch_embedding 的输出特征维度。
    def __init__(self,num_features,num_patches):
        super(PositionEmbeddingStatic, self).__init__()
        #self.input_size = input_size
        #self.output_size = output_size
        self.num_features = num_features
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, num_features))
        self.pos_drop = nn.Dropout(p=Droprate)

    def forward(self, x):
      #  x=self.PatchEmbedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token, x), dim=1)# [B, 1 + num_patches, num_features]
        # 添加位置嵌入
        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
        x = x + pos_embed
        # 应用 Dropout
        x = self.pos_drop(x)
        return x

# 创建一个示例输入张量
#x = torch.randn(4, 18711,768)
# 示例输入
x = torch.randn(4, 162, 768)

# 创建 PositionEmbedding 模块
num_patches = x.shape[1]  # 18711  162
num_features = x.shape[2] # 768   768
#pos_embed = PositionEmbeddingDynamic(imgH=1600,imgW=3040,num_features=num_features, num_patches=num_patches,patch_size=190)
pos_embed = PositionEmbeddingStatic(num_features=num_features, num_patches=num_patches)
# 前向传播
output = pos_embed(x)
print(f"Position Embedding Output Shape: {output.shape}")  # static输出torch.Size([4, 163, 768])成功



