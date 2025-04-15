import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 num_classes = 7
num_classes = 7

class CustomLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(CustomLoss, self).__init__()
        self.beta = beta  # 平衡参数
        self.background_class=0

    def forward(self, y_pre, y_batch):  #[B,num_anchor=6,7+4 分类预测+边界框预测]
        # 提取分类预测和边界框预测
        c_pre = y_pre[:, :, :num_classes]
        b_pre = y_pre[:, :, num_classes:]
        c_hat = y_batch[:, :, :num_classes]
        b_hat = y_batch[:, :, num_classes:]

        # 分类损失：同时考虑正负样本
        batch_size = c_pre.shape[0]
        classification_losses = []

        for i in range(batch_size):
            # 获取正样本掩码
            positive_mask = c_hat[i].max(dim=1)[0] > 0
            positive_mask = positive_mask & (c_hat[i].argmax(dim=1) != self.background_class)  # 排除背景类
            
            if torch.any(positive_mask):
                # 正样本损失
                pos_loss = F.cross_entropy(
                    c_pre[i][positive_mask],
                    c_hat[i][positive_mask].argmax(dim=1),
                    reduction='sum'
                )

                # 负样本损失（使用困难负样本挖掘）
                negative_mask = ~positive_mask
                if torch.any(negative_mask):
                    # 计算负样本的置信度损失
                    neg_conf = F.softmax(c_pre[i][negative_mask], dim=1)[:,self.background_class]  # 背景类的置信度
                    num_neg = min(3 * torch.sum(positive_mask), torch.sum(negative_mask))  # 控制正负样本比例
                    _, hard_neg_indices = neg_conf.sort(descending=False)
                    hard_neg_indices = hard_neg_indices[:num_neg]

                    neg_loss = F.cross_entropy(
                        c_pre[i][negative_mask][hard_neg_indices],
                        torch.zeros(num_neg, dtype=torch.long, device=c_pre.device),
                        reduction='sum'
                    )

                    classification_losses.append((pos_loss + neg_loss) / (num_neg + torch.sum(positive_mask)))
                else:
                    classification_losses.append(pos_loss / torch.sum(positive_mask))

        # 计算平均分类损失
        L_class = torch.mean(torch.stack(classification_losses)) if classification_losses else torch.tensor(0.0,
                                                                                                            device=y_pre.device)

        # 边界框回归损失：只对正样本计算
        # positive_mask = c_hat.max(dim=2)[0] > 0
            # 修改正样本掩码生成逻辑（关键修复）
        positive_mask = (c_hat.argmax(dim=2) != self.background_class) & (c_hat.max(dim=2)[0] > 0)
        
        if torch.any(positive_mask):
            L_box = F.smooth_l1_loss(
                b_pre[positive_mask],
                b_hat[positive_mask],
                reduction='sum'
            ) / (torch.sum(positive_mask) + 1e-6)
        else:
            L_box = torch.tensor(0.0, device=y_pre.device)

        # 总损失：平衡分类损失和回归损失
        total_loss = L_class + self.beta * L_box
        return total_loss, L_class, L_box

def generate_random_data(batch_size, num_anchors, num_classes):
    # 生成随机的模型输出 y_pre
    y_pre = torch.randn(batch_size, num_anchors, num_classes + 4)

    # 生成随机的真实标签 y_batch
    c_hat = torch.zeros(batch_size, num_anchors, num_classes)
    b_hat = torch.rand(batch_size, num_anchors, 4)

    # 随机选择一些锚框作为缺陷锚框，并设置它们的真实类别
    for i in range(batch_size):
        defect_indices = torch.randint(0, num_anchors, (torch.randint(0, num_anchors, (1,)).item(),))
        c_hat[i, defect_indices] = F.one_hot(torch.randint(0, num_classes, (len(defect_indices),)),
                                                 num_classes).float()

    y_batch = torch.cat((c_hat, b_hat), dim=2)

    return y_pre, y_batch


def test_custom_loss():
    # 设置参数
    batch_size = 4
    num_anchors = 6
    num_classes = 6
    beta = 1.0

    # 生成随机数据
    y_pre, y_batch = generate_random_data(batch_size, num_anchors, num_classes)

    # 实例化 CustomLoss
    criterion = CustomLoss(beta=beta)

    # 计算损失
    loss = criterion(y_pre, y_batch)

    # 打印损失值
    print(f"Computed Loss: {loss.item()}")#Computed Loss: 0.9916955232620239


if __name__ == "__main__":
    test_custom_loss()