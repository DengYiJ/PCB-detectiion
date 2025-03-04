import torch
import torch.nn as nn
from sympy import false

import Pyramid
import PatchEmbedding
import PositionEmbedding
import FeedForward
import Transformer
import SelectiveKernelConv
import FulllyConnectedLayer
import DepthwiseSeparableConvolution
from Transformer import SparseAttention,Block
from param import Netdepth


#from PatchEmbedding import embed_dim, img_width, img_height


class MNT(nn.Module):
    def __init__(self,embed_dim,norm_layer,hideF,num_heads): #imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
        super().__init__()
        self.patch_embedding=PatchEmbedding.PatchEmbedding1(embed_dim=embed_dim,norm_layer=norm_layer)
        self.sparse_attention=SparseAttention(dim=embed_dim,num_heads=num_heads,window_size=64,block_size=32)       # num_patches = self.patch_embedding.num_patches
       # self.position_embedding = PositionEmbedding.PositionEmbeddingDynamic(input_size=num_patches,output_size=num_patches + 1,num_features=embed_dim, num_patches=num_patches)
    #self.position_embedding=PositionEmbedding.PositionEmbeddingStatic(num_features=num_features,num_patches=num_patches)
        self.feedforward=FeedForward.FeedForward(in_features=embed_dim,hidden_features=hideF,out_features=embed_dim,drop=0.1)
        # self.blocks=nn.Sequential(*[
        #     Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=4,window_size=64,block_size=32,drop=0.1,attn_drop=0.1,drop_path=0.1)for _ in range(depth)])
      # self.attention=Transformer.Attention(dim=embed_dim,num_heads=num_heads,qkv_bias=False,attn_drop=0.0,proj_drop=0.0)
       # self.FullyconnectedLayer=FulllyConnectedLayer.FullyConnected()
       # self.DSC=DepthwiseSeparableConvolution.DepthwiseSeparableConvolution()

    def forward(self, input):
        x=self.patch_embedding(input)
        # 动态获取 num_patches 和 num_features
        if x.ndim == 3:
            B, num_patches, num_features = x.shape
            #print(f"num_patches: {num_patches}, num_features: {num_features}")
        else:
            raise ValueError("Unexpected input shape for position embedding")
        # 动态创建 PositionEmbedding
        position_embedding = PositionEmbedding.PositionEmbeddingStatic(num_features=num_features,num_patches=num_patches)
        x = position_embedding(x)
       # x=self.position_embedding(x)
        assert isinstance(x, torch.Tensor), "Position embedding output is not a tensor"
        identity = x.clone()   # x.shape=(B,N,192), identity.shape=(B,N,64)
        x=self.feedforward(x)
        x=self.sparse_attention(x)
        x=x+identity
        identity = x.clone()
        x=self.feedforward(x)
        assert isinstance(x, torch.Tensor), "Position embedding output is not a tensor"
        x=x+identity
        return x


# MNT测试代码
def testMNT():
    # 输入张量 (batch_size=4, channels=3, height=1600, width=3040)
    input = torch.randn(4, 3, 1600, 3040)
    embed_dim = 192
    norm_layer = nn.LayerNorm
    hideF = 256
    num_heads = 4

    # 创建 MNT 模型
    mnt = MNT(embed_dim=embed_dim, norm_layer=norm_layer, hideF=hideF, num_heads=num_heads)

    # 前向传播
    output = mnt(input)
    print(output.shape)
    print(f"MNT模块通过！")#输出torch.Size([4, 1025, 192]) ok将他通过五次，写成一个列表



class pyramid(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.pyramid=Pyramid.pyramid(in_channels=in_channels,out_channels=out_channels)

    def forward(self, x):
        x=self.pyramid(x)
        return x

def testpyramid():  #将列表作为pyramid输入
    input = torch.randn(4, 1025, 192)

class ADD(nn.Module):
    def __init__(self,add_in_channel, M=2, Groups=8, ratio=16, WH=1):
        super().__init__()
        #fully_connected 层的输入特征数 in_features 应等于连接后的特征总数。
        #out_features 是全连接层的输出特征数,随你定
        self.fcl=FulllyConnectedLayer.FullyConnected(in_features=add_in_channel,out_features=128)
        self.skc=SelectiveKernelConv.SKConv(
            features=self.fcl.out_features, # 输入通道数来自 fcl.out_features
            WH=WH,         # 输入特征图的空间维度
            M=M,           # 分支数量（通常取 2 或 3）
            G=Groups,           # 卷积组的数量（通常取 8 或 16）
            r=ratio    )       # 压缩倍数（通常取 8 或 16）


    def forward(self, x):
        x=self.fcl(x)
        # Assuming x is a 2D tensor (batch_size, features), reshape to (batch_size, features, 1, 1)  为了输入进SKC
        x = x.unsqueeze(-1).unsqueeze(-1)
        x=self.skc(x)  #output:（batch_size,features,1,1）
        return x.squeeze() # 压缩多余的维度  （batch_size,features,1,1）


class model(nn.Module):
    def __init__(self,embed_dim,norm_layer,num_heads,hideF,
                 Pyin_channels,Pyout_channels,
                 num_classes,num_anchors,Netdepth):
        super().__init__()
        #self.img_size = img_size
        #self.patch_size = patch_size
        #self.imgH = imgH
        #self.imgW = imgW
        #self.netdepth=Netdepth

        self.mnt=MNT(embed_dim,norm_layer,hideF,num_heads)# MNT返回最终的输出特征x，形状为 (B, N, C)
        self.pyramid=pyramid(Pyin_channels,Pyout_channels)# #返回最终的输出特征list,b1-b6 #特别的，in channels是embed_dim
        #Adin_channels=Pyout_channels*(5*FimgH*FimgW+FimgH*FimgW//4)
        #self.ADD=ADD(Adin_channels,FimgH,FimgW)
        self.num_classes=num_classes
        self.num_anchors=num_anchors
        self.classification_head = nn.Linear(in_features=128, out_features=num_classes*num_anchors)  # 分类头
        self.bbox_head = nn.Linear(in_features=128,out_features=4*num_anchors)  # 边界框回归头

    def forward(self, x):
        mnt_outputs=[]
        for i in range(5):
            # 确保输入 x 是4维张量，例如通过 reshape 或添加维度
            x=self.mnt(x)
            if x.ndim == 3:  # 如果 x 是三维张量
               x = x.unsqueeze(1)  # 添加一个维度，变为四维张量
            #print(f"MNT {i+1} output shape: {x.shape}")
            mnt_outputs.append(x)#attention输出(B,N,C)
            # MNT返回最终的输出特征x，形状为 (B, N, C)，在循环中，变成一个list，含5个(B,N,C)
            # 将 Sequence 转换为 Convolutional 特征图
        #print(f"MNT out_list:{mnt_outputs}")
        #conv_features = []
        #for output in mnt_outputs:
            # B,_, num_patches_plus_1, D = output.shape
            # num_patches = num_patches_plus_1 - 1
            # img_height=self.imgH
            # img_width=self.imgW#是输入图像的尺寸，
            # patch_height = patch_width =self.patch_size# 表示图像被划分为 (H/P, W/P) 个 patches，其中 H 和 W 是图像的高度和宽度，P 是 patch_size。
            # grid_size = (img_height // patch_height, img_width // patch_width)
            # patches = output[:, 1:, :]  # 去掉了 cls_token，只保留 num_patches 个 patches。
            # patches = patches.permute(0, 2, 1)  # (B, D, num_patches),使用 permute 将维度从 (B, N, D) 转换为 (B, D, N)。
            # conv_feature = patches.reshape(B, D, grid_size[0], grid_size[1])  # (B, D, H, W),使用 reshape 将 (B, D, H*W) 转换为 (B, D, H, W)，得到类似卷积特征图的形状。
            # conv_features.append(conv_feature)
            # #conv_feature  一个list，含5个(B,D,H,W)
            # 调用 FPN
            # Convert each MNT output to (B, C, N) for Conv1d
        # 移除多余的维度 1
        mnt_outputs = [output.squeeze(1) for output in mnt_outputs]
        mnt_outputs = [output.permute(0, 2, 1) for output in mnt_outputs]#交换为(B,C,N)
        feature_list = self.pyramid(mnt_outputs)
        #print("feature_list shapes:")
        # for feature in feature_list:
        #     print(feature.shape)
        #feature_list = self.pyramid(conv_features)#返回一个list，含5个(B,MEoutchannels,H,W)，MEoutchannels作为新D，新维度-》(B,MEoutchannels,N)
            #将b1-b6展平并链接
        flattened_features = [torch.flatten(feature, start_dim=1) for feature in feature_list]#每个特征图被展平成一个二维张量，其中第一维度是批量大小（B），第二维度是所有其他维度的乘积。
        concatenated_features = torch.cat(flattened_features, dim=1)#将所有展平后的特征按列（dim=1）拼接起来。拼接后的张量形状为 [2, 96*5 +24]。
                            #concatenated_features->torch.Size([2, 504])
        #print("flattened_features shapes:")
        # for feat in flattened_features:
        #     print(feat.shape)
        #print(f"concatenated_features shape: {concatenated_features.shape}")
            #调用ADD
        #调用concatenated_features的第二维度作为ADD初始化in_channels的形参
        self.ADD=ADD(add_in_channel=concatenated_features.shape[1])
        add_output=self.ADD(concatenated_features)
        #print(f"add_output shape:{add_output.shape}")
        classification_scores = self.classification_head(add_output)  # 分类分数,输入128，输出6
        bbox_predictions = self.bbox_head(add_output)  # 边界框预测位置，输入128，输出4

        # Reshape to (batch_size, num_anchors, num_classes) and (batch_size, num_anchors,4)
        classification_scores = classification_scores.view(-1, self.num_anchors, self.num_classes)
        #print(f"classification_scores shape:{classification_scores.shape}")
        bbox_predictions = bbox_predictions.view(-1, self.num_anchors, 4)
        #print(f"bbox_predictions shape:{bbox_predictions.shape}")

        # Concatenate along the last dimension to form y_pred
        y_pred = torch.cat([classification_scores, bbox_predictions], dim=-1)
        #print(f"y_pred shape:{y_pred.shape}")
        #print(f"y_pre dtype: {y_pred.dtype}")
        y_pred = y_pred.to(torch.float16)#强制转换
        return y_pred


def testmodel():
    input = torch.randn(2, 3, 224, 224)#1600,3040
    embed_dim = 64
    norm_layer = nn.LayerNorm
    hideF = 256
    num_heads = 4
    Pyin_channels = embed_dim
    Pyout_channels = 32
    imgH = 1600
    imgW = 3040
    FimgH = 4
    FimgW = 4
    num_classes = 6
    num_anchors = 6
    netdepth=2
    model_test = model(
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        num_heads=num_heads,
        hideF=hideF,
        Pyin_channels=Pyin_channels,
        Pyout_channels=Pyout_channels,
        # imgH=imgH,
        # imgW=imgW,
        # FimgH=FimgH,
        # FimgW=FimgW,
        num_classes=num_classes,
        num_anchors=num_anchors,
        Netdepth=netdepth
    )

    output = model_test(input)
    print(f"Output shape: {output.shape}")  # 预期输出形状为 (2, 6, 8) 或相似，具体取决于 num_anchors 和 num_classes
    torch.save(model_test, 'test_model.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
   testmodel()
    #testMNT()