import torch
from torch import nn
import torch.nn.functional as F
droprate=0.5

class PositionEmbedding(nn.Module):
    def __init__(self, input_size, output_size,num_features,num_patches):
        super(PositionEmbedding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_features = num_features
        # --------------------------------------------------------------------------------------------------------------------#
        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        # --------------------------------------------------------------------------------------------------------------------#
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        # --------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        # --------------------------------------------------------------------------------------------------------------------#
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, num_features))
        self.pos_drop = nn.Dropout(p=droprate)

    def forward(self, x):
        x=self.PatchEmbedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token, x), dim=1)

        cls_token_pe=self.pos_embed[:,0:1,:]
        img_token_pe=self.pos_embed[:,1:,:]

        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        x = self.pos_drop(x + pos_embed)







