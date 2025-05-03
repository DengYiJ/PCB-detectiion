import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops # Import for NMS

BACKGROUND_CLASS=0
class_names = {
    1: 'missing_hole',
    2: 'mouse_bite',
    3: 'open_circuit',
    4: 'short',
    5: 'spur',
    6: 'spurious_copper'
}

# 颜色映射 (BGR格式)
colors = {
    1: (0, 0, 255),  # 红色
    2: (0, 255, 0),  # 绿色
    3: (255, 0, 0),  # 蓝色
    4: (0, 255, 255),  # 黄色
    5: (255, 0, 255),  # 紫色
    6: (255, 255, 0)  # 青色
}


def convert_padded_norm_to_original_pixel(boxes_padded_norm, original_size, padded_size, scale, pad_x, pad_y):
    """
        将边界框从归一化到填充后尺寸 [0, 1] 的坐标，
        转换为原始图片像素坐标 [xmin, ymin, xmax, ymax]。

        Args:
            boxes_padded_norm (np.ndarray): numpy 数组 (N, 4)，边界框归一化到填充后尺寸 [0, 1]。
            original_size (tuple): 原始图片尺寸 (orig_width, orig_height)。
            padded_size (tuple): 填充后的图片尺寸 (padded_width, padded_height)。
            scale (float): 图片预处理时使用的缩放比例。
            pad_x (int): 图片预处理时在 X 方向的填充量。
            pad_y (int): 图片预处理时在 Y 方向的填充量。

        Returns:
            np.ndarray: numpy 数组 (N, 4)，边界框在原始图片像素坐标系 [x1, y1, x2, y2]。
        """
    orig_w, orig_h = original_size
    padded_w, padded_h = padded_size
    # 确保输入是 numpy 数组
    if isinstance(boxes_padded_norm, torch.Tensor):
        boxes_padded_norm = boxes_padded_norm.cpu().numpy()

    # 处理空输入
    if boxes_padded_norm.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    # 在副本上操作
    boxes_original_pixel = boxes_padded_norm.copy()

    # 1. 从 [0, 1] 归一化恢复到 padded_size 像素坐标
    boxes_original_pixel[:, 0] *= padded_w
    boxes_original_pixel[:, 1] *= padded_h
    boxes_original_pixel[:, 2] *= padded_w
    boxes_original_pixel[:, 3] *= padded_h

    # Remove padding and scale back to original pixel coordinates
    # Ensure scale is not zero to avoid division by zero
    if scale == 0:
        print("Warning: Scale is zero during decoding.")
        return np.zeros_like(boxes_original_pixel)  # Return zero boxes or handle as error

    boxes_original_pixel[:, 0] = (boxes_original_pixel[:, 0] - pad_x) / scale
    boxes_original_pixel[:, 1] = (boxes_original_pixel[:, 1] - pad_y) / scale
    boxes_original_pixel[:, 2] = (boxes_original_pixel[:, 2] - pad_x) / scale
    boxes_original_pixel[:, 3] = (boxes_original_pixel[:, 3] - pad_y) / scale

    # 3. 可选：将坐标裁剪到原始图片边界内 [0, original_size]
    boxes_original_pixel[:, 0] = np.clip(boxes_original_pixel[:, 0], 0, orig_w)
    boxes_original_pixel[:, 1] = np.clip(boxes_original_pixel[:, 1], 0, orig_h)
    boxes_original_pixel[:, 2] = np.clip(boxes_original_pixel[:, 2], 0, orig_w)
    boxes_original_pixel[:, 3] = np.clip(boxes_original_pixel[:, 3], 0, orig_h)

    return boxes_original_pixel  # 返回 numpy 数组，坐标在原始像素系 (x1, y1, x2, y2)
def decode_boxes(anchor_boxes, raw_predictions):
    """
    Decodes bounding box predictions from the model's raw output.

    Args:
        anchor_boxes (torch.Tensor): Anchor boxes in [xmin, ymin, xmax, ymax] format (Num_Anchors, 4).
        raw_predictions (torch.Tensor): Model's raw box predictions in [dx, dy, dw, dh] format (Num_Anchors, 4).

    Returns:
        torch.Tensor: Decoded bounding boxes in [xmin, ymin, xmax, ymax] format (Num_Anchors, 4).
    """

    """
    Decodes bounding box predictions from the model's raw output.
    """
    # --- Debugging: Check tensor devices ---
    # print(f"decode_boxes input - anchor_boxes device: {anchor_boxes.device}")
    # print(f"decode_boxes input - raw_predictions device: {raw_predictions.device}")
    # --- End Debugging ---

    # Convert anchors to [cx, cy, w, h]
    anchor_cx = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
    anchor_cy = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
    anchor_w = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_h = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    # Get predicted offsets and scales
    dx = raw_predictions[:, 0]
    dy = raw_predictions[:, 1]
    dw = raw_predictions[:, 2]
    dh = raw_predictions[:, 3]

    # Decode predicted box center and size
    pred_cx = anchor_cx + dx * anchor_w
    pred_cy = anchor_cy + dy * anchor_h
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)

    # Convert decoded boxes back to [xmin, ymin, xmax, ymax]
    pred_xmin = pred_cx - pred_w / 2
    pred_ymin = pred_cy - pred_h / 2
    pred_xmax = pred_cx + pred_w / 2
    pred_ymax = pred_cy + pred_h / 2

    decoded_boxes = torch.stack([pred_xmin, pred_ymin, pred_xmax, pred_ymax], dim=-1)

    return decoded_boxes
def visualize_predictions(original_images, y_pre, y_batch, output_dir, epoch, num_classes,scales, pad_xs, pad_ys, padded_sizes,final_detections=None):
    """
    在原始图片上绘制预测框和真实框，并将结果保存到文件。

    Args:
        original_images (list): 原始图片的 numpy 数组列表 (H, W, C)。
        y_pre (np.ndarray): 模型的预测结果 (B, num_anchors, num_classes + 4)。
                            边界框 assumed to be normalized [0, 1] relative to padded_size.
        y_batch (np.ndarray): 真实标签 (B, num_anchors, num_classes + 4)。
                             边界框 assumed to be normalized [0, 1] relative to original_size.
        output_dir (str): 保存可视化图片的目录。
        epoch (int): 当前的 epoch 数。
        num_classes (int): 总类别数 (包含背景)。
        scales (list): Batch 中每张图片计算出的缩放比例列表。
        pad_xs (list): Batch 中每张图片计算出的 x 方向填充量列表。
        pad_ys (list): Batch 中每张图片计算出的 y 方向填充量列表。
        padded_sizes (list): Batch 中每张图片填充后的尺寸列表 (通常是 [(1024, 1024), ...])。
        final_detections (list, optional): 经过NMS和过滤后的最终检测结果列表。
                                          格式为 List[List[Dict]]，外层 List 对应 Batch 中的图片，
                                          内层 List 包含字典 {'box': [xmin, ymin, xmax, ymax], 'score': float, 'class_id': int}，
                                          其中 box 坐标是归一化到 padded_size [0, 1] 的。
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(original_images)):
        # 复制原始图片以便在其上绘制
        # original_images 是 numpy 数组列表 (H x W x C)
        image = original_images[i].copy()
        orig_height, orig_width = image.shape[:2]
        # print(f"orig_height:{orig_height}, orig_width:{orig_width}")
        # 计算resize时的缩放比例和padding
        scale_i = scales[i]
        pad_x_i = pad_xs[i]
        pad_y_i = pad_ys[i]
        padded_width_i, padded_height_i = padded_sizes[i]
        # padded_size_i = (padded_width_i, padded_height_i)

        # --- 绘制真实框 (Ground Truth) ---
        # 遍历当前图片的真实标签 (来自 y_batch)
        # for anchor in range(y_batch[i].shape[0]):
        #     class_probs_gt = y_batch[i, anchor, :num_classes]  # 真实的类别 One-Hot (包含背景)
        #     bbox_raw_gt = y_batch[i, anchor, num_classes:]  # 真实的边界框 ( assumed 0-1 in padded_size)
        #     class_idx_gt = np.argmax(class_probs_gt)  # 真实的类别索引
        #
        #     # 跳过背景类 (class_idx=0) 和无效标注 (One-Hot 值 <= 0)
        #     if class_idx_gt == 0 or class_probs_gt[class_idx_gt] <= 0:
        #         continue

        # 绘制预测框
        for anchor in range(y_pre[i].shape[0]):
            # print(f"start drawing anchor {anchor}。")
            # print(f"y_pre[i].")
            # # y_pre[i] 是当前图片的预测结果 (NumPy 数组)
            # 切片结果是 NumPy 数组切
            class_probs_pred = y_pre[i, anchor, :num_classes] # 预测的类别概率 (包含背景)
            bbox_raw_pred = y_pre[i, anchor, num_classes:] # 预测的边界框 ( assumed 0-1 in padded_size)
            class_idx_pred = np.argmax(class_probs_pred) # 预测的类别索引
            score_pred = class_probs_pred[class_idx_pred] # 预测类别的置信度
            # print(f"class_probs_pred: {class_probs_pred}。")
            # print(f"bbox_raw_pred: {bbox_raw_pred}。")
            # print(f"class_idx_pred: {class_idx_pred}。")
            # print(f"score_pred: {score_pred}。")

            if class_idx_pred == 0 or score_pred<=0.5:# 过滤掉背景类预测和低置信度预测
                continue

            if score_pred > 0.5:  # 0.5
                # --- 解码预测框坐标 ---
                # 将预测框从 padded_size 的归一化坐标 [0, 1] 转换回原始图片像素坐标
                # 使用与 MyDataset 中 convert_to_model_input 相反的步骤
                # 1. 从 [0, 1] 归一化坐标恢复到 padded_size 像素坐标
                xmin_pred_padded_pixel = bbox_raw_pred[0] * padded_width_i
                ymin_pred_padded_pixel = bbox_raw_pred[1] * padded_height_i
                xmax_pred_padded_pixel = bbox_raw_pred[2] * padded_width_i
                ymax_pred_padded_pixel = bbox_raw_pred[3] * padded_height_i

                # 2. 减去填充偏移量
                xmin_pred_scaled_pixel = xmin_pred_padded_pixel - pad_x_i
                ymin_pred_scaled_pixel = ymin_pred_padded_pixel - pad_y_i
                xmax_pred_scaled_pixel = xmax_pred_padded_pixel - pad_x_i
                ymax_pred_scaled_pixel = ymax_pred_padded_pixel - pad_y_i

                # 3. 除以缩放比例，恢复到原始图片像素坐标
                # 确保缩放比例不为零
                if scale_i == 0:
                    print(f"警告: 图片 {i} 的缩放比例为零，无法解码预测框。")
                    continue  # 跳过此图片预测框绘制

                xmin_pred_original_pixel = xmin_pred_scaled_pixel / scale_i
                ymin_pred_original_pixel = ymin_pred_scaled_pixel / scale_i
                xmax_pred_original_pixel = xmax_pred_scaled_pixel / scale_i
                ymax_pred_original_pixel = ymax_pred_scaled_pixel / scale_i

                # 确保坐标在原始图片范围内 [0, orig_size]
                xmin_pred_original_pixel = np.clip(xmin_pred_original_pixel, 0, orig_width)
                ymin_pred_original_pixel = np.clip(ymin_pred_original_pixel, 0, orig_height)
                xmax_pred_original_pixel = np.clip(xmax_pred_original_pixel, 0, orig_width)
                ymax_pred_original_pixel = np.clip(ymax_pred_original_pixel, 0, orig_height)

                # 确保坐标是整数类型，用于 OpenCV 绘图
                xmin, ymin, xmax, ymax = int(xmin_pred_original_pixel), int(ymin_pred_original_pixel), int(
                    xmax_pred_original_pixel), int(ymax_pred_original_pixel)

                # 确保边界框有效 (宽度和高度 > 0)
                if xmax <= xmin or ymax <= ymin:
                    # print(f"警告: 图片 {i} 的预测框解码后无效: [{xmin}, {ymin}, {xmax}, {ymax}]。跳过绘制。")
                    continue  # 跳过绘制无效边界框

                # 获取预测类别的颜色
                pred_color = colors.get(class_idx_pred, (255, 255, 255))  # 默认白色

                # 获取预测类别的名称
                class_name_pred = class_names.get(class_idx_pred, f"unknown_{class_idx_pred}")

                # 绘制预测框和标签
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), pred_color, 2)
                cv2.putText(image, f"{class_name_pred}: {score_pred:.2f}",
                            (xmin, ymin - 10),  # 标签位置在框上方
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2)

                # --- 绘制真实框 ---
                # 遍历当前图片的所有 Anchor 的真实标签
        for anchor in range(y_batch[i].shape[0]):
                    # y_batch[i] 是当前图片的真实标签 (NumPy 数组)
                    # 切片结果是 NumPy 数组切片
            class_probs_gt = y_batch[i, anchor, :num_classes]  # 真实的类别 One-Hot (包含背景)
            bbox_raw_gt = y_batch[i, anchor,num_classes:]  # 真实的边界框 ( assumed 0-1 in padded_size) <--- IMPORTANT: Corrected assumption
            class_idx_gt = np.argmax(class_probs_gt)  # 真实的类别索引
            # print(f"class_probs_gt: {class_probs_gt}。")
            # print(f"bbox_raw_gt: {bbox_raw_gt}。")
            # print(f"class_idx_gt: {class_idx_gt}。")
            # 跳过背景类 (class_idx=0) 和无效标注 (One-Hot 值 <= 0)
            if class_idx_gt == 0 or class_probs_gt[class_idx_gt] <= 0:
                continue
            # print(f"\n--- Debugging GT Anchor {anchor} in Image {i} ---")
            # print("GT Class Index:", class_idx_gt)
            # print("GT Class Prob:", class_probs_gt[class_idx_gt])  # Use NumPy array
            # print("Raw GT bbox (0-1 padded):", bbox_raw_gt)  # Use NumPy array
            # --- 解码真实框坐标 ---
            # 将真实框从 padded_size 的归一化坐标 [0, 1] 转换回原始图片像素坐标
            # 使用与预测框相同的反向转换逻辑

            # 1. 从 [0, 1] 归一化坐标恢复到 padded_size 像素坐标
            xmin_gt_padded_pixel = bbox_raw_gt[0] * padded_width_i
            ymin_gt_padded_pixel = bbox_raw_gt[1] * padded_height_i
            xmax_gt_padded_pixel = bbox_raw_gt[2] * padded_width_i
            ymax_gt_padded_pixel = bbox_raw_gt[3] * padded_height_i
            # print("GT bbox (padded pixel float):", xmin_gt_padded_pixel, ymin_gt_padded_pixel,
            #               xmax_gt_padded_pixel, ymax_gt_padded_pixel)
            # 2. 减去填充偏移量
            xmin_gt_scaled_pixel = xmin_gt_padded_pixel - pad_x_i
            ymin_gt_scaled_pixel = ymin_gt_padded_pixel - pad_y_i
            xmax_gt_scaled_pixel = xmax_gt_padded_pixel - pad_x_i
            ymax_gt_scaled_pixel = ymax_gt_padded_pixel - pad_y_i
            # print("GT bbox (scaled pixel float):", xmin_gt_scaled_pixel, ymin_gt_scaled_pixel,
            #               xmax_gt_scaled_pixel, ymax_gt_scaled_pixel)
                    # 3. 除以缩放比例，恢复到原始图片像素坐标
                    # 确保缩放比例不为零
            if scale_i == 0:
                print(f"警告: 图片 {i} 的缩放比例为零，无法解码真实框。")
                continue  # 跳过此图片真实框绘制

            xmin_gt_original_pixel = xmin_gt_scaled_pixel / scale_i
            ymin_gt_original_pixel = ymin_gt_scaled_pixel / scale_i
            xmax_gt_original_pixel = xmax_gt_scaled_pixel / scale_i
            ymax_gt_original_pixel = ymax_gt_scaled_pixel / scale_i
            # print("GT bbox (original pixel float):", xmin_gt_original_pixel, ymin_gt_original_pixel,
            #               xmax_gt_original_pixel, ymax_gt_original_pixel)

                    # 确保坐标在原始图片范围内 [0, orig_size]
            xmin_gt_original_pixel = np.clip(xmin_gt_original_pixel, 0, orig_width)
            ymin_gt_original_pixel = np.clip(ymin_gt_original_pixel, 0, orig_height)
            xmax_gt_original_pixel = np.clip(xmax_gt_original_pixel, 0, orig_width)
            ymax_gt_original_pixel = np.clip(ymax_gt_original_pixel, 0, orig_height)
            # print("GT bbox (original pixel float after clip):", xmin_gt_original_pixel, ymin_gt_original_pixel,
            #               xmax_gt_original_pixel, ymax_gt_original_pixel)

                    # 确保坐标是整数类型，用于 OpenCV 绘图
            xmin, ymin, xmax, ymax = int(xmin_gt_original_pixel), int(ymin_gt_original_pixel), int(
                        xmax_gt_original_pixel), int(ymax_gt_original_pixel)

            # print("Decoded GT bbox (original pixel float):", xmin_gt_original_pixel, ymin_gt_original_pixel,
            #               xmax_gt_original_pixel, ymax_gt_original_pixel)
            # print("Decoded GT bbox (original pixel int):", xmin, ymin, xmax, ymax)

                    # 确保边界框有效 (宽度和高度 > 0)
            if xmax <= xmin or ymax <= ymin:
            # print(f"警告: 图片 {i} 的真实框解码后无效: [{xmin}, {ymin}, {xmax}, {ymax}]。跳过绘制。")
                print("Warning: Decoded GT box is invalid. Skipping drawing.")
                continue  # 跳过绘制无效边界框
            else:
                print("Decoded GT box is valid. Attempting to draw.")
                # print("--- End Debugging GT Anchor ---")
            # 获取真实类别的颜色 (通常使用一个固定的颜色，如红色或蓝色，以便与预测框区分)
            gt_color = (0, 0, 255)  # lan色 (RGB 格式)

                    # 获取真实类别的名称
            class_name_gt = class_names.get(class_idx_gt, f"unknown_{class_idx_gt}")
            # print("Attempting to draw GT box...")
                    # 绘制真实框和标签
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), gt_color, 2)
            cv2.putText(image, f"{class_name_gt} (GT)",  # 标签加上 (GT) 以区分
                                (xmin, ymin - 15),  # 标签位置在框上方，略高于预测标签
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
            # print("GT box drawing attempted.")
                # --- 保存图片 ---
        output_path = os.path.join(output_dir, f"epoch_{epoch}_image_{i}.jpg")
                # OpenCV 的 imwrite 期望 BGR 格式，如果图片是 RGB 需要转换
        if image.shape[-1] == 3:  # 检查是否是彩色图片
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)
        # print(f"Image saved to {output_path}")
def visualize_predictions_new(original_images, y_pre, y_batch, anchor_boxes_xyxy, output_dir, epoch, num_classes, scales, pad_xs, pad_ys, padded_sizes, confidence_threshold=0.7, nms_threshold=0.3):
    """
    在原始图片上绘制模型预测框和真实框，并将结果保存到文件。
    模型预测和真实标签的边界框部分均假定为相对于 Anchor Boxes 的 [dx, dy, dw, dh] 格式。

    Args:
        original_images (list): 原始图片的 numpy 数组列表 (H, W, C)。
        y_pre (torch.Tensor): 模型的预测结果 (B, num_anchors, num_classes + 4)。
                              边界框部分为预测的 [dx, dy, dw, dh] 偏移量。
        y_batch (torch.Tensor): 真实标签 (B, num_anchors, num_classes + 4)。
                                边界框部分为真实的 [gt_dx, gt_dy, gt_dw, gt_dh] 偏移量。
        anchor_boxes_xyxy (torch.Tensor): Anchor boxes in [xmin, ymin, xmax, ymax] format (Num_Anchors, 4)，
                                         归一化到 padded_size [0, 1]。
        output_dir (str): 保存可视化图片的目录。
        epoch (int): 当前的 epoch 数。
        num_classes (int): 总类别数 (包含背景)。
        scales (list): Batch 中每张图片计算出的缩放比例列表。
        pad_xs (list): Batch 中每张图片计算出的 x 方向填充量列表。
        pad_ys (list): Batch 中每张图片计算出的 y 方向填充量列表。
        padded_sizes (list): Batch 中每张图片填充后的尺寸列表 (通常是 [(1024, 1024), ...])。
        confidence_threshold (float): 过滤低置信度预测框的阈值。
        nms_threshold (float): NMS 的 IoU 阈值。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 确保 Anchor Boxes 在 CPU 上进行后续 NumPy 操作
    anchor_boxes_xyxy_cpu = anchor_boxes_xyxy.cpu()


    for i in range(len(original_images)):
        # 复制原始图片以便在其上绘制，转换为 BGR 格式
        image = original_images[i].copy()
        if image.shape[-1] == 3 and image.shape[-1] != 3: # Avoids converting grayscale or alpha images
             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        orig_height, orig_width = image.shape[:2]

        # 获取当前图片对应的预处理参数
        scale_i = scales[i]
        pad_x_i = pad_xs[i]
        pad_y_i = pad_ys[i]
        padded_size_i = padded_sizes[i]
        padded_width_i, padded_height_i = padded_size_i
        original_size_i = (orig_width, orig_height)

        # 获取当前图片的预测结果和真实标签 (移动到 CPU 并转换为 NumPy)
        y_pre_i_cpu = y_pre[i].detach().cpu()
        y_batch_i_cpu = y_batch[i].detach().cpu()

        pred_cls_logits_i = y_pre_i_cpu[:, :num_classes]
        pred_bbox_offsets_i = y_pre_i_cpu[:, num_classes:] # 预测的 [dx, dy, dw, dh]

        gt_class_one_hot_i = y_batch_i_cpu[:, :num_classes]
        gt_bbox_offsets_i = y_batch_i_cpu[:, num_classes:] # 真实的 [gt_dx, gt_dy, gt_dw, gt_dh]

        # 计算预测类别的置信度和索引
        pred_conf_i = torch.softmax(pred_cls_logits_i, dim=-1)
        conf_of_predicted_class_i, pred_cls_idx_i = torch.max(pred_conf_i, dim=1) # (Num_Anchors,)


        # --- 处理和绘制预测框 ---
        # 解码预测偏移量为 padded_norm [xmin, ymin, xmax, ymax]
        # decode_boxes 期望输入是 PyTorch 张量，且在同一设备。Anchor boxes 已经在 CPU (NumPy)， Predictions 在 CPU (Tensor)。
        # 将 Anchor Boxes 转换为 Tensor 以便与 Predictions 一起传递给 decode_boxes
        anchor_boxes_xyxy_i_tensor = anchor_boxes_xyxy_cpu # (Num_Anchors, 4) Tensor on CPU

        decoded_pred_boxes_padded_norm_i = decode_boxes(
             anchor_boxes_xyxy_i_tensor, # Anchor boxes on CPU
             pred_bbox_offsets_i # Predicted offsets on CPU
        ).numpy() # Decode on CPU, convert to NumPy

        # 过滤低置信度预测框 (跳过背景类)
        foreground_pred_mask = (pred_cls_idx_i != BACKGROUND_CLASS) & (conf_of_predicted_class_i > confidence_threshold)
        foreground_pred_indices = torch.where(foreground_pred_mask)[0].numpy() # Get indices as NumPy array

        if len(foreground_pred_indices) > 0:
            filtered_pred_boxes_padded_norm = decoded_pred_boxes_padded_norm_i[foreground_pred_indices]
            filtered_pred_scores = conf_of_predicted_class_i[foreground_pred_indices].numpy()
            filtered_pred_classes = pred_cls_idx_i[foreground_pred_indices].numpy()

            # 转换为原始像素坐标用于 NMS
            filtered_pred_boxes_original_pixel = convert_padded_norm_to_original_pixel(
                filtered_pred_boxes_padded_norm,
                original_size_i, padded_size_i, scale_i, pad_x_i, pad_y_i
            )

            # 对过滤后的预测框应用 NMS (逐类别进行)
            final_pred_boxes_original_pixel = []
            final_pred_scores = []
            final_pred_classes = []

            for class_id in range(num_classes):
                if class_id == BACKGROUND_CLASS: continue

                class_mask = (filtered_pred_classes == class_id)
                if not np.any(class_mask): continue

                boxes_for_nms = filtered_pred_boxes_original_pixel[class_mask]
                scores_for_nms = filtered_pred_scores[class_mask]
                classes_for_nms = filtered_pred_classes[class_mask]

                # Apply NMS
                # Ensure inputs to nms are PyTorch Tensors on CPU
                keep_indices = ops.nms(
                    torch.tensor(boxes_for_nms, dtype=torch.float32),
                    torch.tensor(scores_for_nms, dtype=torch.float32),
                    nms_threshold
                ).numpy() # Get kept indices as NumPy array

                final_pred_boxes_original_pixel.extend(boxes_for_nms[keep_indices])
                final_pred_scores.extend(scores_for_nms[keep_indices])
                final_pred_classes.extend(classes_for_nms[keep_indices])

            # 绘制最终预测框
            for j in range(len(final_pred_boxes_original_pixel)):
                xmin, ymin, xmax, ymax = final_pred_boxes_original_pixel[j].astype(np.int32)
                score = final_pred_scores[j]
                class_id = final_pred_classes[j]
                class_name = class_names.get(class_id, f"unknown_{class_id}")
                color = colors.get(class_id, (255, 255, 255)) # Default white

                # Ensure bounding box is valid before drawing
                if xmax > xmin and ymax > ymin:
                     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                     cv2.putText(image, f"{class_name}: {score:.2f}",
                                 (xmin, ymin - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    print(f"Warning: Invalid predicted box coordinates after NMS: [{xmin}, {ymin}, {xmax}, {ymax}]. Skipping drawing.")


        # --- 处理和绘制真实框 ---
        # 找到所有正样本 Anchor (真实类别不是背景)
        gt_class_indices_i = torch.argmax(gt_class_one_hot_i, dim=1)
        positive_gt_mask = (gt_class_indices_i != BACKGROUND_CLASS) & (gt_class_one_hot_i.max(dim=1)[0] == 1.0) # Ensure it's a valid one-hot for foreground
        positive_gt_indices = torch.where(positive_gt_mask)[0].numpy() # Get indices as NumPy array

        if len(positive_gt_indices) > 0:
            # 获取正样本的真实框偏移量和对应的 Anchor Boxes
            positive_gt_bbox_offsets = gt_bbox_offsets_i[positive_gt_indices]
            positive_anchor_boxes_xyxy = anchor_boxes_xyxy_cpu[positive_gt_indices] # Corresponding Anchor Boxes on CPU

            # 解码真实框偏移量为 padded_norm [xmin, ymin, xmax, ymax]
            # decode_boxes expects PyTorch Tensors
            decoded_gt_boxes_padded_norm = decode_boxes(
                torch.tensor(positive_anchor_boxes_xyxy, dtype=torch.float32), # Anchor boxes as Tensor on CPU
                torch.tensor(positive_gt_bbox_offsets, dtype=torch.float32) # GT offsets as Tensor on CPU
            ).numpy() # Decode on CPU, convert to NumPy

            # 转换为原始像素坐标用于绘制
            decoded_gt_boxes_original_pixel = convert_padded_norm_to_original_pixel(
                 decoded_gt_boxes_padded_norm,
                 original_size_i, padded_size_i, scale_i, pad_x_i, pad_y_i
            )

            # 获取正样本的真实类别索引 (用于标签)
            positive_gt_classes_idx = gt_class_indices_i[positive_gt_indices].numpy()

            # 绘制真实框
            gt_color = (0, 255, 0)  # 绿色
            for j in range(len(decoded_gt_boxes_original_pixel)):
                xmin, ymin, xmax, ymax = decoded_gt_boxes_original_pixel[j].astype(np.int32)
                class_id = positive_gt_classes_idx[j]
                class_name = class_names.get(class_id, f"unknown_{class_id}")

                 # Ensure bounding box is valid before drawing
                if xmax > xmin and ymax > ymin:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), gt_color, 2)
                    cv2.putText(image, f"{class_name} (GT)",
                                (xmin, ymin - 15), # 标签位置略高于预测标签
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
                else:
                    print(f"Warning: Invalid ground truth box coordinates after decoding: [{xmin}, {ymin}, {xmax}, {ymax}]. Skipping drawing.")


        # --- 保存图片 ---
        output_path = os.path.join(output_dir, f"epoch_{epoch}_image_{i}.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved visualized image to {output_path}")
# --- Test routine for visualize_predictions ---
def test_visualize_predictions():
    """
    Test routine to create dummy data and call visualize_predictions.
    Creates dummy data that mimics the output of the DataLoader
    when using the corrected MyDataset and custom_collate_fn.
    """
    print("--- Running visualize_predictions Test Routine ---")

    # Define dummy parameters
    batch_size = 2
    num_anchors = 6
    num_classes = 7 # Including background
    padded_size = (1024, 1024)
    visualizations_dir='./PCB_DATASET/Visualizations'
    epoch = 1

    # Create dummy original images (list of numpy arrays HxWx3)
    original_images = []
    original_sizes = []
    scales = []
    pad_xs = []
    pad_ys = []
    padded_sizes_list = [] # Renamed to avoid conflict

    # Image 1: Original (800, 600)
    orig_w1, orig_h1 = 800, 600
    original_size1 = (orig_w1, orig_h1)
    original_image1 = np.random.randint(0, 256, size=(orig_h1, orig_w1, 3), dtype=np.uint8)
    original_images.append(original_image1)
    original_sizes.append(original_size1)
    # Calculate scale and padding for Image 1 (short side 600 -> 1024)
    scale1 = 1024 / min(orig_w1, orig_h1) # 1024 / 600
    new_w1, new_h1 = int(orig_w1 * scale1), int(orig_h1 * scale1) # 1365, 1024
    pad_x1 = (padded_size[0] - new_w1) // 2 # (1024 - 1365) // 2 = -170
    pad_y1 = (padded_size[1] - new_h1) // 2 # (1024 - 1024) // 2 = 0
    scales.append(scale1)
    pad_xs.append(pad_x1)
    pad_ys.append(pad_y1)
    padded_sizes_list.append(padded_size) # All padded to 1024x1024

    # Image 2: Original (1200, 1500)
    orig_w2, orig_h2 = 1200, 1500
    original_size2 = (orig_w2, orig_h2)
    original_image2 = np.random.randint(0, 256, size=(orig_h2, orig_w2, 3), dtype=np.uint8)
    original_images.append(original_image2)
    original_sizes.append(original_size2)
    # Calculate scale and padding for Image 2 (short side 1200 -> 1024)
    scale2 = 1024 / min(orig_w2, orig_h2) # 1024 / 1200
    new_w2, new_h2 = int(orig_w2 * scale2), int(orig_h2 * scale2) # 1024, 1280
    pad_x2 = (padded_size[0] - new_w2) // 2 # (1024 - 1024) // 2 = 0
    pad_y2 = (padded_size[1] - new_h2) // 2 # (1024 - 1280) // 2 = -128
    scales.append(scale2)
    pad_xs.append(pad_x2)
    pad_ys.append(pad_y2)
    padded_sizes_list.append(padded_size)


    # Create dummy y_batch (NumPy array B x Anchors x (Classes + Boxes))
    # Boxes are 0-1 in padded_size (1024, 1024)
    y_batch = np.zeros((batch_size, num_anchors, num_classes + 4), dtype=np.float32)

    # Image 1 GTs (Class 1, Class 2)
    # Anchor 0: Class 1 at padded [0.4, 0.4, 0.6, 0.6]
    y_batch[0, 0, 1] = 1.0 # Class 1
    y_batch[0, 0, num_classes:] = [0.4, 0.4, 0.6, 0.6]
    # Anchor 1: Class 2 at padded [0.7, 0.7, 0.9, 0.9]
    y_batch[0, 1, 2] = 1.0 # Class 2
    y_batch[0, 1, num_classes:] = [0.7, 0.7, 0.9, 0.9]
    # Anchors 2-5: Background (already 0)
    y_batch[0, 2:, BACKGROUND_CLASS] = 1.0 # Explicitly set background for others

    # Image 2 GTs (Class 3)
    # Anchor 0: Class 3 at padded [0.2, 0.2, 0.3, 0.3]
    y_batch[1, 0, 3] = 1.0 # Class 3
    y_batch[1, 0, num_classes:] = [0.2, 0.2, 0.3, 0.3]
    # Anchors 1-5: Background (already 0)
    y_batch[1, 1:, BACKGROUND_CLASS] = 1.0 # Explicitly set background for others


    # Create dummy y_pre (NumPy array B x Anchors x (Classes + Boxes))
    # Boxes are 0-1 in padded_size (1024, 1024)
    y_pre = np.zeros((batch_size, num_anchors, num_classes + 4), dtype=np.float32)

    # Image 1 Predictions (near GTs)
    # Anchor 0: Class 1 prediction near GT Anchor 0, high confidence
    y_pre[0, 0, 1] = 0.95 # Confidence for Class 1
    y_pre[0, 0, num_classes:] = [0.41, 0.41, 0.61, 0.61] # Prediction box
    # Anchor 1: Class 2 prediction near GT Anchor 1, high confidence
    y_pre[0, 1, 2] = 0.90 # Confidence for Class 2
    y_pre[0, 1, num_classes:] = [0.71, 0.71, 0.91, 0.91] # Prediction box
    # Anchor 2: Class 1 prediction, low confidence (should be filtered)
    y_pre[0, 2, 1] = 0.4 # Confidence
    y_pre[0, 2, num_classes:] = [0.5, 0.5, 0.7, 0.7] # Prediction box
    # Anchors 3-5: Background predictions (high confidence for background)
    y_pre[0, 3:, BACKGROUND_CLASS] = 0.99 # Confidence for Background

    # Image 2 Predictions (near GTs)
    # Anchor 0: Class 3 prediction near GT Anchor 0, high confidence
    y_pre[1, 0, 3] = 0.92 # Confidence for Class 3
    y_pre[1, 0, num_classes:] = [0.21, 0.21, 0.31, 0.31] # Prediction box
    # Anchors 1-5: Background predictions
    y_pre[1, 1:, BACKGROUND_CLASS] = 0.99 # Confidence for Background


    # Call the visualize_predictions function
    visualize_predictions(
        original_images=original_images,
        y_pre=y_pre,
        y_batch=y_batch,
        output_dir=visualizations_dir,
        epoch=epoch,
        num_classes=num_classes,
        scales=scales,
        pad_xs=pad_xs,
        pad_ys=pad_ys,
        padded_sizes=padded_sizes_list
    )

    print("--- visualize_predictions Test Routine Finished ---")
    print(f"Check the '{visualizations_dir}' directory for output images.")


# Add the main block to run the test routine
if __name__ == "__main__":
    # Need to define BACKGROUND_CLASS, class_names, colors if not imported globally
    # Assuming they are defined globally in this file or imported
    # Example definitions if needed:
    # BACKGROUND_CLASS = 0
    # class_names = {1: 'class1', 2: 'class2', 3: 'class3', ...}
    # colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), ...}

    test_visualize_predictions()