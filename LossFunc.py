import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 num_classes = 6
num_classes = 6

class CustomLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(CustomLoss, self).__init__()
        self.beta = beta  # 平衡参数
    ''''
    def forward(self, y_pre, y_batch):
        # y_pre: 模型输出，形状为 (batch_size, num_anchors, num_classes + 4)
        # y_batch: 真实标签，形状为 (batch_size, num_anchors, num_classes + 4)

        # 提取分类预测和边界框预测
        c_pre = y_pre[:, :, :num_classes]  # 分类预测 (batch_size, num_anchors, num_classes)
        b_pre = y_pre[:, :, num_classes:]  # 边界框预测 (batch_size, num_anchors, 4)

        # 提取分类标签和边界框标签
        c_hat = y_batch[:, :, :num_classes]  # 分类标签 (batch_size, num_anchors, num_classes)
        b_hat = y_batch[:, :, num_classes:]  # 边界框标签 (batch_size, num_anchors, 4)

        # 确保分类标签和预测的形状正确
        assert c_pre.shape == c_hat.shape, f"Shape mismatch: c_pre {c_pre.shape} vs c_hat {c_hat.shape}"
        # 计算分类损失 L_class
        # 使用交叉熵损失
        batch_size, num_anchors, _ = c_pre.shape
        classification_losses = []
        for i in range(batch_size):
            defect_mask = c_hat[i].argmax(dim=1).bool()  # 获取缺陷锚框的掩码 (num_anchors,)
            if torch.any(defect_mask):
                class_loss = F.cross_entropy(c_pre[i][defect_mask], c_hat[i][defect_mask].argmax(dim=1),
                                             reduction='sum')
                classification_losses.append(class_loss)

        if classification_losses:
            L_class = sum(classification_losses) / batch_size
        else:
            L_class = torch.tensor(0.0, device=y_pre.device)

        # 计算边界框回归损失 L_box
        # 仅对包含缺陷的锚框（c_hat_j = 1）计算
        defect_mask = c_hat.argmax(dim=2).bool()  # 获取缺陷锚框的掩码 (batch_size, num_anchors)
        if torch.any(defect_mask):  # 如果存在缺陷锚框
            L_box = F.smooth_l1_loss(b_pre[defect_mask].float(), b_hat[defect_mask].float(),
                                     reduction='sum') / batch_size
        else:
            L_box = torch.tensor(0.0, device=y_pre.device)  # 如果没有缺陷锚框，损失为 0

        # 总损失
        total_loss = (L_class + self.beta * L_box) / (torch.sum(defect_mask) + 1e-8)
        return total_loss
        '''''

    def forward(self, y_pre, y_batch):
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
                    neg_conf = F.softmax(c_pre[i][negative_mask], dim=1)[:, 0]  # 背景类的置信度
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
        positive_mask = c_hat.max(dim=2)[0] > 0
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