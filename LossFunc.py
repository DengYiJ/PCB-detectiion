import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 num_classes = 6
num_classes = 6

class CustomLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(CustomLoss, self).__init__()
        self.beta = beta  # 平衡参数

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
        L_class = F.cross_entropy(c_pre.transpose(1, 2), c_hat.argmax(dim=2), reduction='mean')

        # 计算边界框回归损失 L_box
        # 仅对包含缺陷的锚框（c_hat_j = 1）计算
        defect_mask = c_hat.argmax(dim=2).bool()  # 获取缺陷锚框的掩码 (batch_size, num_anchors)
        if torch.any(defect_mask):  # 如果存在缺陷锚框
            L_box = F.smooth_l1_loss(b_pre[defect_mask].float(), b_hat[defect_mask].float(), reduction='mean')
        else:
            L_box = torch.tensor(0.0, device=y_pre.device)  # 如果没有缺陷锚框，损失为 0

        # 总损失
        total_loss = L_class + self.beta * L_box
       # total_loss = total_loss.to(torch.float32)  # 强制转换
        return total_loss