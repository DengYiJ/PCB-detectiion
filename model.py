import torch
import torch.nn as nn
import Pyramid
import PatchEmbedding
import PositionEmbedding
import FeedForward
import Transformer
import SelectiveKernelConv
import FulllyConnectedLayer
import DepthwiseSeparableConvolution
from PatchEmbedding import embed_dim, img_width


class MNT(nn.Module):
    def __init__(self,imgH,imgW,patch_size,in_channels,embed_dim,norm_layer,num_heads): #imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
        super().__init__()
        self.patch_embedding=PatchEmbedding.PatchEmbedding(img_size=(imgH,imgW),patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim,norm_layer=norm_layer)
        num_patches = self.patch_embedding.num_patches
       # self.position_embedding = PositionEmbedding.PositionEmbeddingDynamic(input_size=num_patches,output_size=num_patches + 1,num_features=embed_dim, num_patches=num_patches)
        self.position_embedding=PositionEmbedding.PositionEmbeddingStatic(num_features=embed_dim,num_patches=num_patches)
        self.feedforward=FeedForward.FeedForward(in_features=embed_dim,hidden_features=None,out_features=embed_dim,drop=0.1)
        self.attention=Transformer.Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=False,attn_drop=0.0,proj_drop=0.0)
       # self.FullyconnectedLayer=FulllyConnectedLayer.FullyConnected()
       # self.DSC=DepthwiseSeparableConvolution.DepthwiseSeparableConvolution()

    def forward(self, input):
        x=self.patch_embedding(input)
        x=self.position_embedding(x)
        identity=x
        x=self.feedforward(x)
        x=self.attention(x)
        x+=identity
        identity=x
        x=self.feedforward(x)
        x+=identity

class pyramid(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.pyramid=Pyramid.pyramid(in_channels=in_channels,out_channels=out_channels)

    def forward(self, x):
        x=self.pyramid(x)
        return x

class ADD(nn.Module):
    def __init__(self,in_channel,imgH,imgW,num_pic=6, M=2, Groups=8, ratio=16, WH=128):
        super().__init__()
        #fully_connected 层的输入特征数 in_features 应等于连接后的特征总数。
        #out_features 是全连接层的输出特征数,随你定
        self.fcl=FulllyConnectedLayer.FullyConnected(in_features=in_channel,out_features=128)
        self.skc=SelectiveKernelConv.SKConv(
            features=self.fcl.out_features, # 输入通道数来自 fcl.out_features
            WH=WH,         # 输入特征图的空间维度
            M=M,           # 分支数量（通常取 2 或 3）
            G=Groups,           # 卷积组的数量（通常取 8 或 16）
            r=ratio    )       # 压缩倍数（通常取 8 或 16）

    def forward(self, x):
        x=self.fcl(x)
        # Assuming x is a 2D tensor (batch_size, features), reshape to (batch_size, features, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x=self.skc(x)
        return x.squeeze() # 压缩多余的维度


class model(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embed_dim,norm_layer,num_heads,
                 Pyin_channels,Pyout_channels,
                 Adin_channels,imgH,imgW,num_pic,M,Groups,ratio,WH,
                 num_classes,num_anchors):
        super().__init__()
        self.mnt=MNT(img_size,patch_size,in_channels,embed_dim,norm_layer,num_heads)# MNT返回最终的输出特征x，形状为 (B, N, C)
        self.pyramid=pyramid(Pyin_channels,Pyout_channels)# #返回最终的输出特征list,b1-b6 #特别的，in channels是embed_dim
        self.ADD=ADD(Adin_channels,imgH,imgW)
        self.num_classes=num_classes
        self.num_anchors=num_anchors
        self.classification_head = nn.Linear(128, num_classes*num_anchors)  # 分类头
        self.bbox_head = nn.Linear(128, 4*num_anchors)  # 边界框回归头
        self.img_size=img_size
        self.patch_size=patch_size
        self.imgH=imgH
        self.imgW=imgW

    def forward(self, x):
        mnt_outputs=[]
        for i in range(5):
            x=self.mnt(x)
            mnt_outputs.append(x)#attention输出(B,N,C)
            # MNT返回最终的输出特征x，形状为 (B, N, C)，在循环中，变成一个list，含5个(B,N,C)
            # 将 Sequence 转换为 Convolutional 特征图

        conv_features = []
        for output in mnt_outputs:
            B, num_patches_plus_1, D = output.shape
            num_patches = num_patches_plus_1 - 1
            img_height=self.imgH
            img_width=self.imgW#是输入图像的尺寸，
            patch_height = patch_width =self.patch_size# 表示图像被划分为 (H/P, W/P) 个 patches，其中 H 和 W 是图像的高度和宽度，P 是 patch_size。
            grid_size = (img_height // patch_height, img_width // patch_width)
            patches = output[:, 1:, :]  # 去掉了 cls_token，只保留 num_patches 个 patches。
            patches = patches.permute(0, 2, 1)  # (B, D, num_patches),使用 permute 将维度从 (B, N, D) 转换为 (B, D, N)。
            conv_feature = patches.reshape(B, D, grid_size[0], grid_size[1])  # (B, D, H, W),使用 reshape 将 (B, D, H*W) 转换为 (B, D, H, W)，得到类似卷积特征图的形状。
            conv_features.append(conv_feature)
            #conv_feature  一个list，含5个(B,D,H,W)
            # 调用 FPN
        feature_list = self.pyramid(conv_features)
            #将b1-b6展平并链接
        flattened_features = [torch.flatten(feature, start_dim=1) for feature in feature_list]
        concatenated_features = torch.cat(flattened_features, dim=1)

            #调用ADD
        add_output=self.ADD(concatenated_features)

        classification_scores = self.classification_head(add_output)  # 分类分数,输入128，输出6
        bbox_predictions = self.bbox_head(add_output)  # 边界框预测位置，输入128，输出4

        # Reshape to (batch_size, num_anchors, num_classes) and (batch_size, num_anchors,4)
        classification_scores = classification_scores.view(-1, self.num_anchors, self.num_classes)
        bbox_predictions = bbox_predictions.view(-1, self.num_anchors, 4)

        # Concatenate along the last dimension to form y_pred
        y_pred = torch.cat([classification_scores, bbox_predictions], dim=-1)

        return y_pred
