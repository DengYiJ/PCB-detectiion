import torch
import torch.nn as nn
from sklearn.metrics.pairwise import additive_chi2_kernel
from sympy import false
from torch import autocast

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

class MNT(nn.Module):
    def __init__(self,embed_dim,norm_layer,hideF,num_heads ,device='cuda'): #imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
        super().__init__()
        self.patch_embedding=PatchEmbedding.PatchEmbedding1(embed_dim=embed_dim,norm_layer=norm_layer).to(device)
        self.sparse_attention=SparseAttention(dim=embed_dim,num_heads=num_heads,window_size=64,block_size=32).to(device)    # num_patches = self.patch_embedding.num_patches
        self.feedforward=FeedForward.FeedForward(in_features=embed_dim,hidden_features=hideF,out_features=embed_dim,drop=0.1).to(device)
        self.device = device

    @autocast('cuda')
    def forward(self, input):
        print("input device:", input.device)
        x=self.patch_embedding(input)
        # 动态获取 num_patches 和 num_features
        if x.ndim == 3:
            B, num_patches, num_features = x.shape
            #print(f"num_patches: {num_patches}, num_features: {num_features}")
        else:
            raise ValueError("Unexpected input shape for position embedding")
        # 动态创建 PositionEmbedding
        position_embedding = PositionEmbedding.PositionEmbeddingStatic(num_features=num_features,num_patches=num_patches).to(self.device)
        # x = position_embedding(x.to(position_embedding.cls_token.device))#确保 x 被移动到了与 cls_token 相同的设备上
        x = position_embedding(x)
        #print("x device:", x.device)
        #print(f"posEmbed output tensor dtype: {x.dtype}")#posEmbed output tensor dtype: torch.float32
       # x=self.position_embedding(x)
        assert isinstance(x, torch.Tensor), "Position embedding output is not a tensor"
        identity = x.clone()   # x.shape=(B,N,192), identity.shape=(B,N,64)
        identity = identity.to(input.device)
        print("identity device:", identity.device)
        x=self.feedforward(x)
        x=self.sparse_attention(x)
        x=x+identity
        identity = x.clone()
        identity = identity.to(input.device)
        x=self.feedforward(x)
        assert isinstance(x, torch.Tensor), "feedforward embedding output is not a tensor"
        x=x+identity
        #print(f"mnt output tensor dtype: {x.dtype}")
        #转换为16位
       # x=x.half()
        #print(f"mnt output1 tensor dtype: {x.dtype}")
        print("x type:", type(x))
        print("x shape:", x.shape)
        return x


# MNT测试代码
def testMNT():
    # 输入张量 (batch_size=4, channels=3, height=1600, width=3040)
    input = torch.randn(4, 3, 1600, 3040)
    input1=torch.randn(4, 3, 1600, 3040).cuda()
    embed_dim = 192
    norm_layer = nn.LayerNorm
    hideF = 256
    num_heads = 4

    # 创建 MNT 模型
    mnt = MNT(embed_dim=embed_dim, norm_layer=norm_layer, hideF=hideF, num_heads=num_heads).cuda()

    # 前向传播
    output = mnt(input1)
#    print(output.shape)
    print(f"MNT模块通过！")#输出torch.Size([4, 1025, 192]) ok将他通过五次，写成一个列表


class pyramid(nn.Module):
    def __init__(self,in_channels,out_channels,device='cuda'):
        super().__init__()
        self.device=device
        self.pyramid=Pyramid.pyramid(in_channels=in_channels,out_channels=out_channels).to(self.device)

    @autocast('cuda')
    def forward(self, x):
       # print("input device:", x.device)
        x=self.pyramid(x)
        return x

def testpyramid():  #将列表作为pyramid输入
   # input = torch.randn(4, 1025, 192).cuda()
    # 初始化 pyramid 模型
    # 设置超参数
    B = 2  # 批处理大小
    D = 3  # 输入通道数（in_channels）
    N= 64 # 特征图的高度和宽度
    in_channels = D
    out_channels = 6  # 输出通道数
    MEin_channels=out_channels
    MEout_channels=out_channels
    # 生成随机输入特征图列表
    features = []
    for _ in range(5):
        feature = torch.randn(B, D, N).cuda() # 输入带有5个(B, D, H, W)的list
        features.append(feature)
    for i, feature in enumerate(features):
        print(f"Feature {i + 1} shape: {feature.shape}, dtype: {feature.dtype}")
    #print(features)
    # 初始化金字塔模块
    pyramid_model = pyramid(in_channels, out_channels).cuda()
    pyramid_model.half()
    # 测试前向传播
    output_features = pyramid_model(features)


    for i, feature in enumerate(output_features):
        print(f"Feature {i + 1} shape: {feature.shape}, dtype: {feature.dtype}")

    for name, param in pyramid_model.named_parameters():
        print(f"Parameter '{name}' dtype: {param.dtype}")
    # print(f"Pyramid模块通过")
    # print(f"Output features shapes:")
    # for i, feature in enumerate(output_features):
    #     print(f"Feature {i + 1} shape: {feature.shape}")#测试通过

class ADD(nn.Module):
    def __init__(self,add_in_channel, M=2, Groups=8, ratio=16, WH=1,device='cuda'):
        super().__init__()
        #fully_connected 层的输入特征数 in_features 应等于连接后的特征总数。
        #out_features 是全连接层的输出特征数,随你定
        self.device = device
        self.fcl=FulllyConnectedLayer.FullyConnected(in_features=add_in_channel,out_features=128).to(self.device)
        self.skc=SelectiveKernelConv.SKConv(
            features=self.fcl.out_features, # 输入通道数来自 fcl.out_features
            WH=WH,         # 输入特征图的空间维度
            M=M,           # 分支数量（通常取 2 或 3）
            G=Groups,           # 卷积组的数量（通常取 8 或 16）
            r=ratio    ) .to(self.device)      # 压缩倍数（通常取 8 或 16）


    @autocast('cuda')
    def forward(self, x):
        x=self.fcl(x)
        # Assuming x is a 2D tensor (batch_size, features), reshape to (batch_size, features, 1, 1)  为了输入进SKC
        x = x.unsqueeze(-1).unsqueeze(-1)
        x=self.skc(x)  #output:（batch_size,features,1,1）
        return x.squeeze() # 压缩多余的维度  （batch_size,features,1,1）

def testADD():
    # 设置超参数
    batch_size = 2
    add_in_channel = 1024  # 输入通道数
    M = 2  # 分支数量
    Groups = 8  # 卷积组的数量
    ratio = 16  # 压缩倍数
    WH = 1  # 输入特征图的空间维度

    # 生成随机输入张量
    input_tensor = torch.randn(batch_size, add_in_channel).cuda()

    # 初始化 ADD 模型
    add_model = ADD(add_in_channel, M, Groups, ratio, WH,device='cuda').cuda()
#模型取半
    add_model.half()
    # 测试前向传播
    with autocast('cuda'):
        output = add_model(input_tensor)

    print(f"ADD模块通过")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    # 打印模型参数的数据类型
    for name, param in add_model.named_parameters():
        print(f"Parameter '{name}' dtype: {param.dtype}")

class model(nn.Module):
    def __init__(self,embed_dim,norm_layer,num_heads,hideF,
                 Pyin_channels,Pyout_channels,
                 num_classes,num_anchors,Netdepth,device='cuda'):
        super().__init__()
        self.device = device
        self.mnt=MNT(embed_dim,norm_layer,hideF,num_heads).to(self.device)# MNT返回最终的输出特征x，形状为 (B, N, C)
        self.pyramid=pyramid(Pyin_channels,Pyout_channels).to(self.device) #返回最终的输出特征list,b1-b6 #特别的，in channels是embed_dim
        self.num_classes=num_classes
        self.num_anchors=num_anchors
        self.classification_head = nn.Linear(in_features=128, out_features=num_classes*num_anchors).to(self.device)  # 分类头
        self.bbox_head = nn.Linear(in_features=128,out_features=4*num_anchors).to(self.device)  # 边界框回归头

    def forward(self, x):
        mnt_outputs=[]
        for i in range(5):
            # 确保输入 x 是4维张量，例如通过 reshape 或添加维度
            x=self.mnt(x)
            # 调试：检查 x 的类型
            print("x type:", type(x))
      #      print("x shape:", x.shape)

            if x.ndim == 3:  # 如果 x 是三维张量
               x = x.unsqueeze(1)  # 添加一个维度，变为四维张量
            #print(f"MNT {i+1} output shape: {x.shape}")
            mnt_outputs.append(x)#attention输出(B,N,C)

        mnt_outputs = [output.squeeze(1) for output in mnt_outputs]
        mnt_outputs = [output.permute(0, 2, 1) for output in mnt_outputs]#交换为(B,C,N)
        feature_list = self.pyramid(mnt_outputs)
        # 打印 feature_list 中每个张量的数据类型
        for i, feature in enumerate(feature_list):
            print(f"feature_list[{i}] tensor dtype: {feature.dtype}")
        flattened_features = [torch.flatten(feature, start_dim=1) for feature in feature_list]#每个特征图被展平成一个二维张量，其中第一维度是批量大小（B），第二维度是所有其他维度的乘积。
        concatenated_features = torch.cat(flattened_features, dim=1)#将所有展平后的特征按列（dim=1）拼接起来。拼接后的张量形状为 [2, 96*5 +24]。
       #concatenated_features->torch.Size([2, 504])
        #print("flattened_features shapes:")
        # for feat in flattened_features:
        #     print(feat.shape)
        #print(f"concatenated_features shape: {concatenated_features.shape}")
            #调用ADD
        #调用concatenated_features的第二维度作为ADD初始化in_channels的形参
        self.ADD=ADD(add_in_channel=concatenated_features.shape[1]).to(self.device)
        add_output=self.ADD(concatenated_features)# 输出FP32,估计是因为3个FP16和2个FP32相加，最后形成FP32了
        #print(f"add_output shape:{add_output.shape}")
        print(f"add_output tensor dtype: {add_output.dtype}")
        add_output = add_output.half()  # 转化为FP32作为分类头的输入
        classification_scores = self.classification_head(add_output) # 分类分数,输入128，输出6
        bbox_predictions = self.bbox_head(add_output)  # 边界框预测位置，输入128，输出4
        # Reshape to (batch_size, num_anchors, num_classes) and (batch_size, num_anchors,4)
        classification_scores = classification_scores.view(-1, self.num_anchors, self.num_classes)
        # print(f"classification_scores shape:{classification_scores.shape}")
        bbox_predictions = bbox_predictions.view(-1, self.num_anchors, 4)
        # print(f"bbox_predictions shape:{bbox_predictions.shape}")

        # Concatenate along the last dimension to form y_pred
        y_pred = torch.cat([classification_scores, bbox_predictions], dim=-1)
        #print(f"y_pred shape:{y_pred.shape}")
        #print(f"y_pre dtype: {y_pred.dtype}")
        #y_pred = y_pred.to(torch.float16)#强制转换
        return y_pred


def testmodel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input = torch.randn(2, 3, 224, 224).to(device)#1600,3040
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
        Netdepth=netdepth,
        device='cuda'
    )
    model_test.half()
    output = model_test(input).to(device)
    print(f"Output shape: {output.shape}")  # 预期输出形状为 (2, 6, 8) 或相似，具体取决于 num_anchors 和 num_classes
    torch.save(model_test, 'test_model.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
   testmodel()
    #testMNT()
    #testpyramid()
    #testADD()