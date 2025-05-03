import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 VisualizedPredict.py 导入 decode_boxes 函数
from VisualizedPredict import decode_boxes

# 假设 num_classes = 7 (确保与您的 param.py 或实际类别数一致)
# num_classes = 7 # 这里可以从 param 中导入，或者确保一致性
# BACKGROUND_CLASS = 0 # 假设背景类别索引是 0

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # 修改这里以兼容 Python 3，并确保 alpha 是 Tensor
        if isinstance(alpha,(float,int)):
             self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
             self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # input: Logits tensor, shape (N, C) where N is num_samples, C is num_classes
        # target: True class index tensor, shape (N,)

        # 确保输入维度正确
        if input.dim() > 2:
             # This part seems intended for inputs like (N, C, H, W) or (N, C, D1, D2...)
             # If your input is already (N, C), this reshaping might not be needed or should be adapted.
             # Assuming input is already (N, C) from CustomLoss.forward
             # input = input.view(input.size(0),input.size(1),-1)
             # input = input.transpose(1,2)
             # input = input.contiguous().view(-1,input.size(2))
             pass # Assuming input is already (N, C)

        # Ensure target is (N,)
        if target.dim() == 1:
             target = target.unsqueeze(1) # Shape (N, 1)
        elif target.dim() != 2 or target.shape[1] != 1:
             raise ValueError(f"Target must be shape (N,) or (N, 1), got {target.shape}")


        logpt = F.log_softmax(input, dim=1) # Apply log_softmax along the class dimension
        # logpt shape: (N, C)

        # Gather log probabilities for the true class
        # target shape: (N, 1)
        logpt = logpt.gather(1, target) # Result shape (N, 1)
        logpt = logpt.squeeze(1) # Result shape (N,) - log probability for the true class


        # pt = Variable(logpt.data.exp()) # Remove Variable
        pt = logpt.exp() # Convert log probability to probability


        if self.alpha is not None:
            # Ensure alpha tensor is on the correct device and dtype
            if self.alpha.device != input.device or self.alpha.dtype != input.dtype:
                 self.alpha = self.alpha.to(input.device, dtype=input.dtype)

            # Gather alpha weight for the true class
            # target shape: (N, 1)
            at = self.alpha.gather(0, target.view(-1)) # target.view(-1) is (N,)
            # at shape: (N,)

            # Apply alpha weight
            logpt = logpt * at


        # Focal Loss core calculation
        # loss shape: (N,) - individual loss for each sample
        loss = -1 * (1 - pt)**self.gamma * logpt


        # Handle reduction (mean or sum)
        if self.size_average:
            # If size_average is True, return the mean loss over samples
            return loss.mean()
        else:
            # If size_average is False, return the tensor of individual losses
            return loss

class CustomLoss(nn.Module):
    # 添加 anchor_boxes_xyxy 参数到初始化方法
    def __init__(self, num_classes, anchor_boxes_xyxy, beta=0.5, neg_pos_ratio=3,gamma=2.0, alpha=None):
        super(CustomLoss, self).__init__()
        self.beta = beta  # 平衡参数
        self.background_class = 0 # 假设背景类别的索引是 0
        # 负样本与正样本数量的比例。例如，如果找到 N 个正样本，就选择最多 N * neg_pos_ratio 个难负样本。
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes

        # 存储 Anchor Boxes，并在 forward 中将其移动到与输入相同的设备上
        # 假设 anchor_boxes_xyxy 形状是 (Num_Anchors, 4)
        self.register_buffer('anchor_boxes_xyxy', anchor_boxes_xyxy) # 使用 register_buffer 存储非训练参数
        # 实例化 Focal Loss 用于分类损失
        if alpha is None:
            # 或者更简单，只设置一个较低的背景 alpha，其他类的 alpha 默认为 1.0 (FocalLoss.__init__ 会处理)
            if num_classes > 1:
                alpha_list  = [0.25] + [1.0] * (num_classes - 1) # 例如
                print(f"Using default alpha for Focal Loss: {alpha_list}")
                alpha = alpha_list
            else:
                # 如果只有背景一个类，alpha 设置为 [1.0]
                alpha = [1.0]
                print("Warning: num_classes is 1 (only background). Alpha set to [1.0].")

        self.classification_loss_func = FocalLoss(gamma=gamma, alpha=alpha, size_average=False)  # 可以根据需要传入 gamma 和 alpha
        # alpha 参数可以是一个列表，对应每个类别的权重
        # 回归损失将直接在 forward 方法中计算 IoU (1 - IoU)
    def forward(self, y_pre, y_batch):  #[B,num_anchor,num_classes+4 分类预测+边界框预测]
        """
        计算自定义损失。

        Args:
            y_pre (torch.Tensor): 模型输出，形状 (batch_size, num_anchors, num_classes + 4)。 dx dy dw dh
            y_batch (torch.Tensor): 真实标签，形状 (batch_size, num_anchors, num_classes + 4)。dx dy dw dh

        Returns:
            tuple: 总损失，分类损失，回归损失。
        """
        # 提取分类预测和边界框预测
        # 假设 c_pre 是模型的原始输出 (Logits)，未经 Softmax 或 Sigmoid
        c_pre = y_pre[:, :, :self.num_classes]
        # 假设 b_pre 是模型的原始边界框预测 [dx, dy, dw, dh]
        b_pre_raw_regs = y_pre[:, :, self.num_classes:]

        # 提取真实类别 (One-Hot) 和边界框 (xyxy 归一化)
        c_hat = y_batch[:, :, :self.num_classes]
        b_hat_raw_regs  = y_batch[:, :, self.num_classes:] #  真实框目标是 [gt_dx, gt_dy, gt_dw, gt_dh]

        batch_size = c_pre.shape[0]
        num_anchors = c_pre.shape[1]


        # --- 将张量展平以便在整个 Batch 上处理 ---
        c_pre_flat = c_pre.view(-1, self.num_classes) # (Batch*Num_Anchors, num_classes)
        b_pre_raw_regs_flat = b_pre_raw_regs.view(-1, 4) # (Batch*Num_Anchors, 4)
        c_hat_flat = c_hat.view(-1, self.num_classes) # (Batch*Num_Anchors, num_classes)
        b_hat_raw_regs_flat = b_hat_raw_regs .view(-1, 4) # (Batch*Num_Anchors, 4)

        # 为整个 Batch 复制 Anchor Boxes 并展平
        anchor_boxes_xyxy_flat = self.anchor_boxes_xyxy.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, 4)


        # --- 确定正样本和负样本掩码 (在展平的张量上) ---
        # 获取正样本掩码: 真实类别不是背景 (argmax != 0) 且 One-Hot 编码的最大值 > 0 (确保不是全零)
        # 需要确保 c_hat_flat 上的 argmax 操作是针对 One-Hot 编码
        # positive_mask_flat = (c_hat_flat.argmax(dim=1) != self.background_class) & (c_hat_flat.max(dim=1)[0] == 1.0)
        # 更精确的正样本判断：真实类别不是背景 AND 该位置的真实标签在对应类别索引上是 1.0
        gt_class_ids_flat = c_hat_flat.argmax(dim=1)
        # 获取每个位置在其真实类别索引上的 One-Hot 值
        one_hot_value_at_gt_class_flat = torch.gather(
            c_hat_flat, 1, gt_class_ids_flat.unsqueeze(1)
        ).squeeze(1)
        # 正样本掩码: 真实类别不是背景 AND 真实类别位置的值是 1.0 (区分忽略样本)
        # positive_mask_flat = (gt_class_ids_flat != self.background_class) & (one_hot_value_at_gt_class_flat == 1.0)
        # 正样本掩码: 真实类别不是背景 AND 真实类别位置的值是 1.0 (区分忽略样本)
        if self.background_class is not None:
            positive_mask_flat = (gt_class_ids_flat != self.background_class) & (one_hot_value_at_gt_class_flat == 1.0)
        else:
            print("Warning: background_class is not defined in CustomLoss. Assuming index 0 is background.")
            positive_mask_flat = (gt_class_ids_flat != 0) & (one_hot_value_at_gt_class_flat == 1.0)

        # 获取所有潜在的负样本掩码: 真实类别是背景 (argmax == 0)
        negative_mask_all_flat = (gt_class_ids_flat == self.background_class)

        # 忽略样本掩码 (可选): 真实类别位置的值不是 1.0 (假设忽略样本的 One-Hot 是全零)
        # 忽略样本不参与损失计算，所以这里只需识别出来，不需要计算损失。
        # ignored_mask_flat = (one_hot_value_at_gt_class_flat != 1.0) & (~positive_mask_flat) & (~negative_mask_all_flat)
        # 确保掩码是 PyTorch 布尔张量，防止出现 'bool' object 没有 'any'/'sum' 等属性的错误
        if not isinstance(positive_mask_flat, torch.Tensor) or positive_mask_flat.dtype != torch.bool:
            positive_mask_flat = torch.tensor(positive_mask_flat, dtype=torch.bool,
                                              device=gt_class_ids_flat.device)
        if not isinstance(negative_mask_all_flat, torch.Tensor) or negative_mask_all_flat.dtype != torch.bool:
            negative_mask_all_flat = torch.tensor(negative_mask_all_flat, dtype=torch.bool,
                                                  device=gt_class_ids_flat.device)

        # 获取正样本索引
        positive_indices = torch.where(positive_mask_flat)[0]

        # --- 计算分类损失 (使用 Focal Loss,包含 Hard Negative Mining) ---

        # 计算所有样本的分类损失 (使用 Focal Loss, size_average=False 返回每个样本损失)
        # Focal Loss 需要 Logits (N, C) 和 真实类别索引 (N,)
        # 我们将所有正样本和所有潜在负样本的 Logits 和 Target 汇集起来计算 Focal Loss
        # 然后再根据 Hard Negative Mining 的结果进行筛选和求和/平均

        # 收集需要计算分类损失的样本的 Logits 和 Target
        # 包括所有正样本和所有潜在负样本 (Hard Negative Mining 会从负样本中选择)
        classification_mask_all = positive_mask_flat | negative_mask_all_flat # 包含所有正样本和潜在负样本
        classification_pre_all = c_pre_flat[classification_mask_all] # 所有相关预测样本的 Logits（包含所有正样本和潜在负样本）
        classification_hat_all = gt_class_ids_flat[classification_mask_all] # 所有相关样本的真实类别索引

        classification_loss = torch.tensor(0.0, device=y_pre.device) # 默认分类损失为 0

        if negative_mask_all_flat.any():
            # 使用 Focal Loss 计算所有相关样本的独立损失
            # classification_losses_all 是一个张量，包含每个样本的 Focal Loss 值
            classification_losses_all = self.classification_loss_func(classification_pre_all,classification_hat_all)
            # classification_losses_all 的形状是 (Num_Relevant_Samples,)
            # 其中 Num_Relevant_Samples = len(positive_indices) + len(torch.where(negative_mask_all_flat)[0])
            # 现在，根据 Hard Negative Mining 的结果来选择哪些损失要用于最终的总分类损失

            # Hard Negative Mining: 从潜在负样本中选择困难负样本
            num_positive = len(positive_indices)
            num_potential_negatives = negative_mask_all_flat.sum().item()

            if num_positive > 0 and num_potential_negatives > 0:# 只要有正样本或负样本就计算分类损失
            # 筛选出潜在负样本的损失值 (在 classification_losses_all 张量中)
                positive_losses_all_classification = classification_losses_all[positive_mask_flat[classification_mask_all]]  # 形状 (Num_Positive,)

            # 筛选出潜在负样本的损失值 (在 classification_losses_all 张量中)
                negative_losses_all_classification = classification_losses_all[negative_mask_all_flat[classification_mask_all]]  # 形状 (Num_Potential_Negatives,)

            # Hard Negative Mining 步骤: 根据损失值对负样本进行排序并选择
                num_hard_negatives = min(num_potential_negatives, int(num_positive * self.neg_pos_ratio))

                if num_hard_negatives > 0:
                # 对潜在负样本损失进行降序排序
                    sorted_negative_losses, sorted_negative_indices = torch.sort(negative_losses_all_classification,
                                                                             descending=True)
                # 选择损失最高的 top K 个负样本的索引
                    hard_negative_indices_in_all_negatives = sorted_negative_indices[:num_hard_negatives]

                # Hard Negative 的分类损失是这些选中损失的平均
                    classification_loss_negative = negative_losses_all_classification[  hard_negative_indices_in_all_negatives].mean()
                else:
                    classification_loss_negative = torch.tensor(0.0, device=y_pre.device)  # 没有 Hard Negative

            # 正样本的分类损失是它们损失的平均
                classification_loss_positive = positive_losses_all_classification.mean() if len(positive_losses_all_classification) > 0 else torch.tensor(0.0, device=y_pre.device)

            # 总分类损失 (正样本 Focal Loss 平均 + Hard Negative Focal Loss 平均)
            # 这是将正样本和 Hard Negative 的平均损失直接相加
                classification_loss = classification_loss_positive + classification_loss_negative
            else: # 没有正样本也没有潜在负样本 (所有都是忽略样本)
                 classification_loss = torch.tensor(0.0, device=y_pre.device)
        else: # 如果一开始就没有需要计算分类损失的样本 (所有都是忽略样本)
             classification_loss = torch.tensor(0.0, device=y_pre.device)

        # --- 计算边界框回归损失 (使用 IoU Loss，只对正样本计算) ---
        regression_loss_batch = torch.tensor(0.0, device=y_pre.device) # 初始化为零，确保在无正样本时也是张量

        if positive_mask_flat.any():
            # 获取正样本的预测回归目标 [dx, dy, dw, dh]
            pred_box_regs_positive_flat = b_pre_raw_regs_flat[positive_mask_flat]

            # 获取正样本的真实框 [gt_dx, gt_dy, gt_dw, gt_dh]
            target_box_regs_positive_flat = b_hat_raw_regs_flat[positive_mask_flat]

            # 获取正样本对应的 Anchor Box [xmin, ymin, xmax, ymax]
            anchor_boxes_xyxy_positive_flat = anchor_boxes_xyxy_flat[positive_mask_flat]

            # 使用 decode_boxes 将预测回归目标解码为预测边界框 [xmin, ymin, xmax, ymax]
            # decode_boxes 函数需要在同一设备上
            decoded_pred_boxes_xyxy_positive_flat = decode_boxes(
                anchor_boxes_xyxy_positive_flat,
                pred_box_regs_positive_flat
            )
            # 使用 decode_boxes 将真实回归目标解码为真实边界框 [xmin, ymin, xmax, ymax]
            # 这一步很重要，我们需要真实的边界框来计算 IoU
            decoded_target_boxes_xyxy_positive_flat = decode_boxes(
                anchor_boxes_xyxy_positive_flat, # 真实的偏移量相对于相同的 Anchor
                target_box_regs_positive_flat # <--- 真实的偏移量
            )

            # --- 计算正样本的成对 IoU ---
            lt = torch.max(decoded_pred_boxes_xyxy_positive_flat[:, :2], decoded_target_boxes_xyxy_positive_flat[:, :2])
            rb = torch.min(decoded_pred_boxes_xyxy_positive_flat[:, 2:], decoded_target_boxes_xyxy_positive_flat[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, 0] * wh[:, 1]

            area1 = (decoded_pred_boxes_xyxy_positive_flat[:, 2] - decoded_pred_boxes_xyxy_positive_flat[:, 0]).clamp(min=0) * \
                    (decoded_pred_boxes_xyxy_positive_flat[:, 3] - decoded_pred_boxes_xyxy_positive_flat[:, 1]).clamp(min=0)
            area2 = (decoded_target_boxes_xyxy_positive_flat[:, 2] - decoded_target_boxes_xyxy_positive_flat[:, 0]).clamp(min=0) * \
                    (decoded_target_boxes_xyxy_positive_flat[:, 3] - decoded_target_boxes_xyxy_positive_flat[:, 1]).clamp(min=0)

            union = area1 + area2 - inter
            iou_values_positive_flat = inter / (union + 1e-8)


            # 计算 IoU 损失 (1 - IoU)
            iou_loss_positive_flat = 1 - iou_values_positive_flat

            # 对正样本的 IoU 损失求平均
            regression_loss_batch = iou_loss_positive_flat.mean()
            # 如果存在正样本，这里的 mean() 操作会生成一个带有 grad_fn 的张量

        else:
            # 如果没有正样本，回归损失为 0。
            # 为了避免 RuntimeError，确保这个零张量是可以追踪梯度的。
            regression_loss_batch = torch.zeros(1, device=y_pre.device, requires_grad=True)


        # --- 总损失 ---
        # 分类损失和回归损失相加
        total_loss = self.beta * classification_loss + (1-self.beta) * regression_loss_batch

        # 返回总损失以及分类和回归的平均损失（这里的平均是 Batch 内的平均）
        # model_train.py 会再对每个 Epoch 的 Batch 损失求平均
        print(f"total_loss:{total_loss.item():.4f}, classification_loss_batch：{classification_loss.item():.4f}, regression_loss_batch：{regression_loss_batch.item():.4f}")

        # 如果分类损失和回归损失都是 requires_grad=True 的零张量 (Batch 没有正样本或 hard negatives)，
        # 它们的和仍然是 requires_grad=True 的零张量，可以进行 backward()。
        return total_loss, classification_loss, regression_loss_batch

# 注意：generate_random_data 和 test_custom_loss 函数需要根据新的 CustomLoss 初始化方式进行调整
# 它们现在需要生成和传递 anchor_boxes_xyxy
def generate_random_data(batch_size, num_anchors, num_classes):
    # 生成随机的模型输出 y_pre (假设是 Logits + dx, dy, dw, dh)
    y_pre = torch.randn(batch_size, num_anchors, num_classes + 4, requires_grad=True) # Ensure y_pre requires grad

    # 生成随机的真实标签 y_batch (One-Hot 类别 + xyxy 归一化真实框) - 通常不需要 grad
    c_hat = torch.zeros(batch_size, num_anchors, num_classes)
    b_hat_xyxy = torch.rand(batch_size, num_anchors, 4)

    # 随机选择一些锚框作为缺陷锚框，并设置它们的真实类别和边界框
    for i in range(batch_size):
        # 随机选择一些索引作为正样本
        # 确保至少有一些正样本，除非 batch_size 或 num_anchors 极小
        num_positive_samples_i = torch.randint(0, num_anchors // 5 + 1, (1,)).item()
        if num_positive_samples_i > 0:
             positive_indices = torch.randperm(num_anchors)[:num_positive_samples_i]
             # 设置这些正样本的真实类别 (非背景)
             c_hat[i, positive_indices] = F.one_hot(torch.randint(1, num_classes, (len(positive_indices),)), num_classes).float()
             # 随机设置这些正样本的真实边界框
             b_hat_xyxy[i, positive_indices] = torch.rand(len(positive_indices), 4)
             # 确保真实框有效 (xmin < xmax, ymin < ymax)
             b_hat_xyxy[i, positive_indices, 2:] = b_hat_xyxy[i, positive_indices, :2] + torch.rand(len(positive_indices), 2) * 0.1 + 0.01

        # 其余样本为背景
        # background_indices 实际上就是未被选为正样本的索引
        all_indices = torch.arange(num_anchors)
        background_mask = torch.ones(num_anchors, dtype=torch.bool)
        if num_positive_samples_i > 0:
             background_mask[positive_indices] = False
        c_hat[i, background_mask, 0] = 1.0 # 背景类别 One-Hot

    y_batch = torch.cat((c_hat, b_hat_xyxy), dim=2)

    # 生成随机的 Anchor Boxes (xyxy 格式) - 在实际使用中这些是固定的，且通常不会随机生成
    # 这里为了测试方便随机生成，实际应该使用 Dataset 中计算的 reference_boxes
    anchor_boxes_xyxy = torch.rand(num_anchors, 4)
    # 确保 xmin < xmax, ymin < ymax
    anchor_boxes_xyxy[:, 2:] = anchor_boxes_xyxy[:, :2] + torch.rand(num_anchors, 2) * 0.1 + 0.01 # 确保宽度和高度大于0


    return y_pre, y_batch, anchor_boxes_xyxy


def test_custom_loss():
    # 设置参数
    batch_size = 4
    num_anchors = 64 # 增加 Anchor 数量以更接近实际情况
    num_classes = 7 # 确保与 LossFunc 中的一致
    beta = 1.0

    # 生成随机数据
    y_pre, y_batch, anchor_boxes_xyxy = generate_random_data(batch_size, num_anchors, num_classes)

    # 实例化 CustomLoss，传入 num_classes 和 anchor_boxes_xyxy
    criterion = CustomLoss(num_classes=num_classes, anchor_boxes_xyxy=anchor_boxes_xyxy, beta=beta)

    # 计算损失
    total_loss, classification_loss, regression_loss = criterion(y_pre, y_batch)

    # 打印损失值
    print(f"\n--- Custom Loss Test Results ---")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {classification_loss.item():.4f}")
    print(f"Regression (IoU) Loss: {regression_loss.item():.4f}")

    # 尝试进行反向传播，检查是否会报错
    try:
        total_loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Error during backward pass: {e}")

    print(f"------------------------------")


if __name__ == "__main__":
    # 注意：在实际训练中，Anchor Boxes 应该从 Dataset 实例中获取，而不是随机生成
    # 这个测试函数用于验证 LossFunc 本身的逻辑和梯度流
    test_custom_loss()