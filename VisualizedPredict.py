import os
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def nms(boxes, scores, iou_threshold=0.5):
    # 按置信度排序
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        # 保留得分最高的框
        keep.append(order[0])

        # 计算IoU
        ious = calculate_iou(boxes[order[0]], boxes[order[1:]])

        # 移除IoU大于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
def visualize_predictions(original_images, y_pre, y_batch, output_dir, epoch, num_classes, original_sizes):
    # 用于在图像上绘制预测的锚框和分类信息，并将结果保存到新文件夹中
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 定义类别名称
    class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

    # 如果图像是张量，将其转换为 numpy 数组
    # if isinstance(images, torch.Tensor):
    #     images = images.cpu().numpy()
    #     images = np.transpose(images, (0, 2, 3, 1))  # 从 (B, C, H, W) 转换为 (B, H, W, C)
    #     images = (images * 255).astype(np.uint8)
    # else:
    #     # 如果图像是 numpy 数组，确保其形状和数据类型正确
    #     images = np.array(images)
    #     if images.dtype != np.uint8:
    #         images = (images * 255).astype(np.uint8)

    # 将预测和标签从张量转换为 numpy 数组
    if isinstance(y_pre, torch.Tensor):
        y_pre = y_pre.cpu().numpy()
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.cpu().numpy()

    for i in range(len(original_images)):
        # 创建图像副本
        image = original_images[i].copy()
        orig_height, orig_width = image.shape[:2]

        # 处理预测框
        for anchor in range(y_pre[i].shape[0]):
            class_probs = y_pre[i, anchor, :num_classes]
            bbox = y_pre[i, anchor, num_classes:]
            class_idx = np.argmax(class_probs)
            score = class_probs[class_idx]

            if score > 0.5:  # 置信度阈值
                # 获取归一化的边界框坐标
                xmin, ymin, xmax, ymax = bbox

                # 修改坐标转换部分
                scale_x = orig_width / 1024.0
                scale_y = orig_height / 1024.0

                # 先反归一化到1024x1024坐标
                xmin = bbox[0] * 1024
                ymin = bbox[1] * 1024
                xmax = bbox[2] * 1024
                ymax = bbox[3] * 1024

                # 反归一化到原始图像尺寸
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_x)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_x)

                # 确保坐标在图像范围内
                xmin = max(0, min(xmin, orig_width))
                ymin = max(0, min(ymin, orig_height))
                xmax = max(0, min(xmax, orig_width))
                ymax = max(0, min(ymax, orig_height))

                # 绘制预测框（绿色）
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"{class_names[class_idx]}: {score:.2f}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 处理真实框
        for anchor in range(y_batch[i].shape[0]):
            class_probs = y_batch[i, anchor, :num_classes]
            bbox = y_batch[i, anchor, num_classes:]
            class_idx = np.argmax(class_probs)

            if class_probs[class_idx] > 0:  # 只绘制有效的真实框
                # 反归一化到原始图像尺寸
                xmin, ymin, xmax, ymax = bbox
                xmin = int(xmin * orig_width)
                ymin = int(ymin * orig_height)
                xmax = int(xmax * orig_width)
                ymax = int(ymax * orig_height)

                # 确保坐标在图像范围内
                xmin = max(0, min(xmin, orig_width))
                ymin = max(0, min(ymin, orig_height))
                xmax = max(0, min(xmax, orig_width))
                ymax = max(0, min(ymax, orig_height))

                # 绘制真实框（蓝色）
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(image, f"{class_names[class_idx]}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 保存结果
        output_path = os.path.join(output_dir, f"epoch_{epoch}_image_{i}.jpg")
        # 如果图像是RGB格式，需要转换为BGR格式再保存
        if image.shape[-1] == 3:  # 检查是否是彩色图像
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)