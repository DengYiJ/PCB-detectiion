import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

output=torch.randn(4,163,768)
# 创建 FeedForward 层
ff = FeedForward(in_features=768, hidden_features=2048, out_features=768)

# 前向传播
ff_output = ff(output)
print(ff_output.shape)  # 输出: torch.Size([4, 163, 768])