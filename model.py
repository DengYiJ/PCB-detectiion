import math

import torch
import torch.nn as nn
from torch import autocast
import Pyramid
import PatchEmbedding
import PositionEmbedding
import FeedForward
import SelectiveKernelConv
import FulllyConnectedLayer
from Transformer import SparseAttention
import torch.nn.functional as F
from local_aggre import LocalAggregation
from LocalPropagation import LocalPropagation
from resnet_34 import ResNet34_FeatureExtractor
from SRA import SpatialReductionAttention
class MNT(nn.Module):
    def __init__(self,embed_dim,norm_layer,hideF,num_heads,device='cuda'): #imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
        super().__init__()
        self.device = device
        # self.sra_dim = sra_dim
        self.patch_embedding=PatchEmbedding.PatchEmbedding1(embed_dim=embed_dim,norm_layer=norm_layer).to(self.device)
        self.patch_embedding4x4=PatchEmbedding.FixedPatchEmbedding4x4(embed_dim=embed_dim,norm_layer=norm_layer).to(self.device)
        self.sparse_attention=SparseAttention(dim=embed_dim,num_heads=num_heads,window_size=64,block_size=16).to(self.device)    # num_patches = self.patch_embedding.num_patches
        self.sra = SpatialReductionAttention(
            dim=embed_dim,
            # dim=self.sra_dim,
            num_heads=num_heads,
            sr_ratio=2  # 可根据特征图大小调整
        ).to(self.device)
        self.feedforward=FeedForward.FeedForward(in_features=embed_dim,hidden_features=hideF,out_features=embed_dim,drop=0.1).to(self.device)
        self.local_agg=LocalAggregation(embed_dim).to(self.device)
        self.local_pro=LocalPropagation(embed_dim).to(self.device)
        # 在 MNT 模块中添加 Batch Normalization
        self.bn1 = nn.BatchNorm1d(embed_dim).to(self.device)
        self.bn2 = nn.BatchNorm1d(embed_dim).to(self.device)
        # 添加形状转换标志
        self.need_reshape = True
        self.precomputed_shapes = {}  # 新增缓存字典
        self.layer_norm1 = nn.LayerNorm(embed_dim).to(self.device)  # 添加层归一化操作
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def _reshape_to_bchw(self, x):
        """将BNC转换为BCHW"""
        B, N, C = x.shape
        if N == 0:
            return x.reshape(B, C, 0, 0)

            # 从缓存获取或计算形状
        # if N not in self.precomputed_shapes:
        if N > 1:  # 确保有CLS token
            x = x[:, 1:, :]  # 去掉第一个token
            N = N - 1
            sqrt_N = int(math.sqrt(N))
            factors = [(i, N // i) for i in range(sqrt_N, 0, -1) if N % i == 0]
            H, W = factors[0] if factors else (sqrt_N, math.ceil(N / sqrt_N))
            self.precomputed_shapes[N] = (H, W)
        # else:
        #     H, W = self.precomputed_shapes[N]

            # 填充和reshape
        if H * W != N:
            x = F.pad(x, (0, 0, 0, H * W - N), "constant", 0)
        return x.transpose(1, 2).reshape(B, C, H, W)

    # @autocast('cuda')
    def forward(self, input):
        # print("input device:", input.device)
        # B, C, H, W = input.shape
        x=self.patch_embedding4x4(input)
        # print(f"pae:{x.shape}")
        identity = x.clone()
        identity = identity.to(input.device)
        # 动态获取 num_patches 和 num_features
        if x.ndim == 3:
            B, num_patches, num_features = x.shape
            #print(f"num_patches: {num_patches}, num_features: {num_features}")
        else:
            raise ValueError("Unexpected input shape for position embedding")
        # x=self.layer_norm1(x)
        #----------先不做本地增强------------------------
        # x= self.local_agg(x)
        # x=identity+x
        # ----------先不做本地增强------------------------
        # 动态创建 PositionEmbedding
        position_embedding = PositionEmbedding.PositionEmbeddingStatic(num_features=num_features,num_patches=num_patches).to(self.device)
        # x = position_embedding(x.to(position_embedding.cls_token.device))#确保 x 被移动到了与 cls_token 相同的设备上
        x = position_embedding(x)
        # print(f"poe:{x.shape}")
        # print(f"After poe tensor dtype: {x.dtype}")
        #print("x device:", x.device)
        #print(f"posEmbed output tensor dtype: {x.dtype}")#posEmbed output tensor dtype: torch.float32
       # x=self.position_embedding(x)
        assert isinstance(x, torch.Tensor), "Position embedding output is not a tensor"
        identity = x.clone()   # x.shape=(B,N,192), identity.shape=(B,N,64) fp32
        identity = identity.to(input.device)
        # print("identity device:", identity.device)
        # x=self.layer_norm1(x)
        x=self.feedforward(x)
        x=x+identity
        # print(f"After ff tensor dtype: {x.dtype}")
        x=self.layer_norm1(x)
        # print(f"lay1:{x.shape}")
        # 分离CLS token和patch tokens
        cls_token = x[:, :1, :]  # 取第一个token作为CLS token
        patch_tokens = x[:, 1:, :]  # 取剩下的1024个patch tokens
        # B, _, C = x.shape
        # 计算patch tokens的空间维度
        N = patch_tokens.shape[1]  # 1024
        H = int(math.sqrt(N))  # 32
        W = H  # 32
        # print(f"W:{W},H:{H}")
        # x=self.sparse_attention(x) #fp16
        x=self.sra(patch_tokens,H,W)
        # 重新组合CLS token和patch tokens
        x = torch.cat([cls_token, patch_tokens], dim=1)
        # print(f"cat:{x.shape}")
        x = self.local_pro(x)
        # print(f"After Sparse tensor dtype: {x.dtype}")
        x=x+identity #FP32
        # print(f"After add tensor dtype: {x.dtype}")
        # x = x.half()
        # print(f"After half tensor dtype: {x.dtype}")
        # # 应用 Batch Normalization
        # for name, param in self.bn1.named_parameters():  # 打印ff模块的模型参数
        # #     print(f"bn1 Parameter '{name}' dtype: {param.dtype}")
        # x = x.permute(0, 2, 1)  # 调整维度顺序以适应 BatchNorm1d
        # x = self.bn1(x)
        # x = x.permute(0, 2, 1)
        # print(f"After bn1 tensor dtype: {x.dtype},tensor dshape:{x.shape}")  #Size([4, 1025, 192])

        identity = x.clone()
        identity = identity.to(input.device)
        # x = self.local_pro(x)
        x= self.feedforward(x)
        assert isinstance(x, torch.Tensor), "feedforward embedding output is not a tensor"
        x=x+identity
        x=self.layer_norm1(x)

        # 如果需要转换形状
        if self.need_reshape:
            x_bchw = self._reshape_to_bchw(x)
            # print("x_bchw shape:", x_bchw.shape)
            return x_bchw
        else:
            # print("x shape:", x.shape)
            return x



# MNT测试代码
def testMNT():
    # 输入张量 (batch_size=4, channels=3, height=1600, width=3040)
    input1=torch.randn(4, 64, 256, 256).cuda()
    embed_dim = 96
    norm_layer = nn.LayerNorm
    hideF = 256
    num_heads = 4

    # 创建 MNT 模型
    mnt = MNT(embed_dim=embed_dim, norm_layer=norm_layer, hideF=hideF, num_heads=num_heads).cuda()
    #mnt.half()
    # 前向传播
    output = mnt(input1)
#    print(output.shape)
    print(f"BNC shape: {output.shape},")
    print(f"half MNT模块通过！")#输出torch.Size([4, 1025, 192]) ok将他通过五次，写成一个列表


class pyramid(nn.Module):
    def __init__(self,in_channels,out_channels,device='cuda'):
        super().__init__()
        self.device=device
        self.pyramid=Pyramid.pyramid(in_channels=in_channels,out_channels=out_channels).to(self.device)
        # self.bn = nn.BatchNorm2d(out_channels)

    # @autocast('cuda')
    def forward(self, x):
       outputs = self.pyramid(x)
       # print("input device:", x.device)
       # 统一数据类型为float32但保持原始形状
       processed_outputs = []
       for out in outputs:
           # 仅转换数据类型，不改变形状
           out = out.to(dtype=torch.float32)
           # 展平为[B, C*H*W]形式
           out = torch.flatten(out, start_dim=1)  # 从[B,C,H,W]变为[B, C*H*W]
           processed_outputs.append(out)
       return x


def testpyramid():
    # 设置测试参数
    batch_size = 2
    in_channels = 64
    out_channels = 32
    height = 16
    width = 16

    # 生成5个不同分辨率的特征图输入
    features = []
    for scale in range(5):
        # 每个特征图的分辨率递减 (16x16, 8x8, 4x4, 2x2, 1x1)
        h = height // (2 ** scale)
        w = width // (2 ** scale)
        feature = torch.randn(batch_size, in_channels, h, w).cuda()
        features.append(feature)

    # 初始化金字塔模块
    pyramid_model = pyramid(in_channels, out_channels).cuda()

    # 打印输入特征图信息
    print("Input features:")
    for i, feat in enumerate(features):
        print(f"Feature {i + 1}: shape={feat.shape}, dtype={feat.dtype}")

    # 前向传播
    outputs = pyramid_model(features)

    # 打印输出特征图信息
    print("\nOutput features:")
    for i, out in enumerate(outputs):
        print(f"Output {i + 1}: shape={out.shape}, dtype={out.dtype}")

    # 验证输出
    assert len(outputs) == 6, "输出特征图数量不正确"
    for out in outputs:
        assert out.shape[0] == batch_size, "批处理维度不正确"
        assert out.shape[1] == out_channels, "输出通道数不正确"

    print("Pyramid模块测试通过！")
    # for name, param in pyramid_model.named_parameters():
    #     print(f"Parameter '{name}' dtype: {param.dtype}")
    # print(f"Pyramid模块通过")
    # print(f"Output features shapes:")
    # for i, feature in enumerate(output_features):
    #     print(f"Feature {i + 1} shape: {feature.shape}")#测试通过

class ADD(nn.Module):
    def __init__(self,add_in_channel, M=2, Groups=8, ratio=16,device='cuda',num_anchors=6,num_classes=7):
        super().__init__()
        #fully_connected 层的输入特征数 in_features 应等于连接后的特征总数。
        #out_features 是全连接层的输出特征数,随你定
        self.device = device
        self.num_anchors=num_anchors
        self.num_classes=num_classes
        # self.fcl=FulllyConnectedLayer.FullyConnected(in_features=add_in_channel,out_features=128).to(self.device)
         # 修改全连接层结构为渐进式降维
        self.fcl = nn.Sequential(
            nn.Linear(add_in_channel, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)  # 最终降到512维
        ).to(device)
        self.target_channels = 16
        self.target_h = 16
        self.target_w = 16
        self.WH = self.target_h #特征图边长
        self.channel_expand = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.ReLU()
        ).to(device)

        self.skc=SelectiveKernelConv.SKConv(
            features=64, # 输入通道数来自 B,C,H,W
            WH=self.WH,         # 输入特征图的空间维度
            M=M,           # 分支数量（通常取 2 或 3）
            G=Groups,           # 卷积组的数量（通常取 8 或 16）
            r=ratio    ) .to(self.device)      # 压缩倍数（通常取 8 或 16）
          # 添加后处理层
        self.post_skc = nn.Sequential(
            nn.BatchNorm2d(64),  # 对SKC输出的64通道做BN
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 [B,64,8,8]->[B,64,1,1]
            nn.Flatten(),        # [B,64]
            nn.Linear(64, 128),  # 进一步特征变换
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(device)
        self.bn = nn.BatchNorm1d(128).to(self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
         # 并行预测头
 # 分类头调整
        self.class_head = nn.Sequential(
            nn.Linear(128, num_anchors * num_classes),
            nn.Softmax(dim=-1) if num_classes > 1 else nn.Sigmoid()
        )
        
        # 回归头调整
        self.bbox_head = nn.Sequential(
            nn.Linear(128, num_anchors * 4),
            nn.Sigmoid()  # 确保坐标在0-1之间
        )

    # @autocast('cuda')
    def forward(self, x):
        # print(f"[ADD] Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")  # 检查输入状态[2](@ref) fp32
        assert not torch.isnan(x).any(), "NaN in ADD input"
        x=self.fcl(x)
        # x = self.channel_reduce(x)       # [B, 128, 1, 1]
        # print(f"[ADD] After fcl: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}")  # 监测全连接层输出范围[1]
        # SKConv 前检查 NaN/Inf
        assert not torch.isnan(x).any(), "NaN detected before SKConv!"
        assert not torch.isinf(x).any(), "Inf detected before SKConv!"
        x = x.view(x.size(0), self.target_channels, self.target_h, self.target_w)
        # 3. 通道扩展
        x = self.channel_expand(x)  # [B, 8, 8, 8] -> [B, 64, 16, 16]
        # for name, param in self.skc.named_parameters():  # 打印ff模块的模型参数
        #     print(f"skc Parameter '{name}' dtype: {param.dtype}")
        # for name, param in self.skc.named_parameters():
        #     print(f"skc Parameter '{name}' dtype: {param.dtype}")
        x=self.skc(x)  #output:（batch_size,features,1,1） #fp16输入  # (8, 256, 14, 14)
        x = self.post_skc(x)  # [B,128]
         # 并行预测
        cls_pred = self.class_head(x)  # [B, num_anchors*num_classes]
        box_pred = self.bbox_head(x)   # [B, num_anchors*4]
        return cls_pred,box_pred

def testADD():
    # 设置超参数
    batch_size = 2
    add_in_channel = 1024  # 输入通道数
    M = 2  # 分支数量
    Groups = 8  # 卷积组的数量
    ratio = 16  # 压缩倍数
    WH = 1  # 输入特征图的空间维度
    num_anchors = 6
    num_classes = 7
    # 生成随机输入张量
    input_tensor = torch.randn(batch_size, add_in_channel).cuda()

    # 初始化 ADD 模型
    add_model = ADD(add_in_channel, M, Groups, ratio, WH,device='cuda', num_anchors=num_anchors,
        num_classes=num_classes).cuda()
#模型取半
    add_model.half()
    # 测试前向传播
    with autocast('cuda'):
        cls_pred, box_pred = add_model(input_tensor)

    # 打印输出shape
    print(f"Classification prediction shape: {cls_pred.shape}")
    print(f"Bounding box prediction shape: {box_pred.shape}")
    print(f"Expected cls shape: [{batch_size}, {num_anchors*num_classes}]")
    print(f"Expected box shape: [{batch_size}, {num_anchors*4}]")

    # 验证输出形状
    assert cls_pred.shape == (batch_size, num_anchors*num_classes), "分类预测形状不正确"
    assert box_pred.shape == (batch_size, num_anchors*4), "边界框预测形状不正确"

    print("ADD模块测试通过！")
    # 打印模型参数的数据类型
    # for name, param in add_model.named_parameters():
    #     print(f"Parameter '{name}' dtype: {param.dtype}")

class model(nn.Module):
    def __init__(self,embed_dim,norm_layer,num_heads,hideF,
                 Pyin_channels,Pyout_channels,
                 num_classes,num_anchors,Netdepth,device='cuda'):
        super().__init__()
        self.device = device
        self.mnt=MNT(embed_dim,norm_layer,hideF,num_heads).to(self.device)# MNT返回最终的输出特征x，形状为 (B, N, C)
        # 替换单个MNT为ModuleList
        self.mnt_modules = nn.ModuleList([
            MNT(embed_dim, norm_layer, hideF, num_heads).to(device)
            for _ in range(5)  # 假设需要处理5个特征图
        ])
        self.pyramid=pyramid(Pyin_channels,Pyout_channels).to(self.device) #返回最终的输出特征list,b1-b6 #特别的，in channels是embed_dim
        self.num_classes=num_classes
        self.num_anchors=num_anchors
        self.downsample_layers = ResNet34_FeatureExtractor().to(device)
        self.classification_head=nn.Sequential(
            # nn.Linear(128, 256),  # 增加特征维度
            nn.Conv2d(Pyout_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加全局平均池化层
            nn.Flatten(),  # 扁平化特征图
            nn.Linear(256, num_anchors * num_classes),
            # nn.Conv2d(256, num_anchors * num_classes, kernel_size=1, padding=0) # 输出 num_classes * num_anchors
            # nn.Sigmoid()  # 确保输出在0-1范围内
            # nn.Unflatten(-1, (num_anchors, num_classes))  # 重塑为 (batch, num_anchors, num_classes)
        ).to(self.device)
        # 修改边界框预测头部
        self.bbox_head = nn.Sequential(
            # nn.Linear(128, 128),
            nn.Conv2d(Pyout_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(256, num_anchors * 4, kernel_size=1, padding=0),  # 输出 4 * num_anchors
            nn.Flatten(),  # 扁平化特征图
            nn.Linear(256, num_anchors * 4),  #
            nn.Sigmoid()  # 确保输出在0-1范围内
        ).to(self.device)

        # 转换权重为float32
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()

        for module in self.bbox_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()

        self.softmax = nn.Softmax(dim=-1)  # 添加 softmax 激活函数

    # @autocast('cuda')
    def forward(self, x):
        resnet34_features = self.downsample_layers(x)#返回列表
        mnt_outputs_bchw = []
        for i, feat in enumerate(resnet34_features):
            # print(f"resnet output shape: {feat.shape}")
            bchw_output = self.mnt_modules[i](feat)
            # print(f"bchw shape: {bchw_output.shape}")
            mnt_outputs_bchw.append(bchw_output)

        feature_list = self.pyramid(mnt_outputs_bchw)
        # 打印 feature_list 中每个张量的数据类型
        # for i, feature in enumerate(feature_list):
        #     print(f"feature_list[{i}] tensor dtype: {feature.dtype},eature_list[{i}] tensor shape:{feature.shape}")
        # flattened_features = [torch.flatten(feature, start_dim=1) for feature in feature_list]#每个特征图被展平成一个二维张量，其中第一维度是批量大小（B），第二维度是所有其他维度的乘积。
        processed_features = []
        for i, feature in enumerate(feature_list):
            B, C, H, W = feature.shape
        
        # 根据特征图尺寸选择降维方式
            if H >= 128:  # 大特征图
            # 先MaxPooling降维到1/4
            #     pooled = F.max_pool2d(feature, kernel_size=4, stride=4)
                pooled = F.adaptive_avg_pool2d(feature, (16, 16))
                # print(f"After maxpool(4x4): {pooled.shape}")
                flattened = torch.flatten(pooled, start_dim=1)
            elif H >= 32:  # 中等特征图
            # 先MaxPooling降维到1/2
            #     pooled = F.max_pool2d(feature, kernel_size=2, stride=2)
                pooled = F.adaptive_avg_pool2d(feature, (8, 8))
                # print(f"After maxpool(2x2): {pooled.shape}")
                flattened = torch.flatten(pooled, start_dim=1)
            else:  # 小特征图
            # 直接FC降维
                flattened = torch.flatten(feature, start_dim=1)
                # print(f"After NOTHING: {pooled.shape}")
            processed_features.append(flattened)
        
        concatenated_features = torch.cat(processed_features, dim=1)#将所有展平后的特征按列（dim=1）拼接起来。拼接后的张量形状为 [2, 96*5 +24]。
        # print(f"concat fea:{concatenated_features.shape}")
        assert not torch.isnan(concatenated_features).any(), "NaN detected before ADD module!"
        # print(f"concat fea:{concatenated_features.shape}")#concat fea:torch.Size([2, 167936])
       #concatenated_features->torch.Size([2, 504])
        #print("flattened_features shapes:")
        # for feat in flattened_features:
        #     print(feat.shape)
        #print(f"concatenated_features shape: {concatenated_features.shape}")
        # print(f"concat tensor dtype: {concatenated_features.dtype}") #concat tensor FP32
            #调用ADD
        #调用concatenated_features的第二维度作为ADD初始化in_channels的形参
        #concatenated_features.half()
        self.ADD=ADD(add_in_channel=concatenated_features.shape[1],num_anchors=self.num_anchors,num_classes=self.num_classes)
        # self.ADD=self.ADD.to(self.device).half()#必须half否则权重为FP32
        self.ADD = self.ADD.to(self.device)
        # for name, param in self.ADD.named_parameters():  # 打印ff模块的模型参数
        #     print(f"Parameter '{name}' dtype: {param.dtype}")
    # #    print(f"concat half dtype: {concatenated_features.half().dtype}")#concat half fp16
        # add_output=self.ADD(concatenated_features)# 输出FP32,估计是因为3个FP16和2个FP32相加，最后形成FP32了
        cls,bbox=self.ADD(concatenated_features)  #Classification prediction shape: torch.Size([2, 42]) Bounding box prediction shape: torch.Size([2, 24])
         # 调整输出形状
        cls = cls.view(-1, self.num_anchors, self.num_classes)  # [B, num_anchors, num_classes]
        bbox = bbox.view(-1, self.num_anchors, 4)              # [B, num_anchors, 4]
        
        # 合并预测结果
        y_pred = torch.cat([cls, bbox], dim=-1)  # [B, num_anchors, num_classes+4]
        return y_pred  #FP16


def testmodel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    input = torch.randn(2, 3,1024,1024).to(device)#1600,3040
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
    num_classes = 7
    num_anchors = 6
    netdepth=2
    model_test = model(
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        num_heads=num_heads,
        hideF=hideF,
        Pyin_channels=Pyin_channels,
        Pyout_channels=Pyout_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        Netdepth=netdepth,
        device='cuda'
    )
    # model_test.half()
    output = model_test(input).to(device)
    # print(f"Output shape: {output.shape}")  # 预期输出形状为 (2, 6, 8) 或相似，具体取决于 num_anchors 和 num_classes
    # torch.save(model_test, 'test_model.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
     testmodel()
    # testMNT()
    # testpyramid()
    # testADD()