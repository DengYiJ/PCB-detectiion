import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast


class FeedForward(nn.Module):#in_features就是embed_dim 768,attention输出torch.randn(4, 163, 768)
    def __init__(self, in_features, hidden_features=256, out_features=None,act_layer=nn.GELU, drop=0.5):
        super(FeedForward, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        drop_probs =(drop,drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

        # # 将权重和偏置转换为 FP16
        # self.fc1.weight = nn.Parameter(self.fc1.weight.to(torch.float16))
        # self.fc1.bias = nn.Parameter(self.fc1.bias.to(torch.float16))
        # self.fc2.weight = nn.Parameter(self.fc2.weight.to(torch.float16))
        # self.fc2.bias = nn.Parameter(self.fc2.bias.to(torch.float16))

    @autocast('cuda')
    def forward(self, x):
        #确保输入是FP16
       # print(f"x tensor dtype: {x.dtype}")
        # x = x.to(self.fc1.weight.device)
        # x=x.type(torch.float16)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        #print(f"After ff tensor dtype: {x.dtype}")
        # 将输出转换为 FP32
       # x = x.to(torch.float32)
        return x
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