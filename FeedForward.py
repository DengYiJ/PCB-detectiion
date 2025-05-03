import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast


class ffn(nn.Module):#in_features就是embed_dim 768,attention输出torch.randn(4, 163, 768)
    def __init__(self, in_features, hidden_features=256, out_features=None,
                 act_layer=nn.GELU, drop=0.5, use_layernorm=True):
        super(ffn, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features

        # 添加输入归一化
        self.norm1 = nn.LayerNorm(in_features) if use_layernorm else nn.Identity()

        # 主干网络
        self.ffn = nn.Sequential(
            # 第一个全连接块
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layernorm else nn.Identity(),
            act_layer(),
            nn.Dropout(drop),

            # 中间层
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layernorm else nn.Identity(),
            act_layer(),
            nn.Dropout(drop),

            # 输出层
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

        # 添加残差连接的门控机制
        self.gate = nn.Parameter(torch.ones(1))

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # @torch.amp.autocast(device_type="cuda")
    def forward(self, x):
        # 应用输入归一化
        # with torch.cuda.amp.autocast():
        x = self.norm1(x)
        # 主干前向传播
        identity = x

        chunk_size = x.shape[1] // 4  # 按序列长度分4块
        x = torch.cat([
            self.ffn(x_chunk)
            for x_chunk in x.split(chunk_size, dim=1)
        ], dim=1)

        # 残差连接with门控
        # x = identity + self.gate * x

        return identity + self.gate * x
def test_ff():
   # output=torch.randn(4,163,768).half()
    input1 = torch.randn(4, 163, 768).cuda()
    # print(f"After input1 tensor dtype: {input1.dtype}")
# 创建 FeedForward 层
    ff = FeedForward(in_features=768, hidden_features=2048, out_features=768).cuda()
    # 将模型转换为 FP16
    ff = ff.half()
# 前向传播
    ff_output = ff(input1)
   # print(ff_output.shape)  # 输出: torch.Size([4, 163, 768])
    #print("Output dtype:", ff_output.dtype)


    # for name, param in ff.named_parameters():
    #     print(f"Parameter '{name}' dtype: {param.dtype}")

if __name__ == "__main__":
    test_ff()