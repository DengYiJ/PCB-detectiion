# -*- coding: utf-8 -*-
import math
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from torch.utils.data import DataLoader
import numpy as np
from param import root_dir, TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, EPOCHS, LEARNING_RATE, Embeding_dim, Netdepth,visualizations_dir, Num_Anchors
import model
import Dataset
import LossFunc
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from VisualizedPredict import visualize_predictions,decode_boxes,class_names
import torch.nn.functional as F
   # 应用 NMS - 使用torchvision.ops.nms
from torchvision.ops import nms
import os
BACKGROUND_CLASS=0
# ✅ 修改 evaluate_model 方法签名，接收 anchor_boxes_xyxy 参数
def evaluate_model(self, validation_dl, model, num_classes, epoch, anchor_boxes_xyxy, iou_threshold=0.5, device='cuda'):
    model.eval()  # Set model to evaluation mode
    # Initialize variables for validation loss calculation
    total_val_loss = 0
    total_val_cls_loss = 0  # Optional: log component losses
    total_val_box_loss = 0  # Optional: log component losses
    # ✅ 实例化 CustomLoss，传入 num_classes 和 anchor_boxes_xyxy
    criterion = LossFunc.CustomLoss(num_classes=num_classes, anchor_boxes_xyxy=anchor_boxes_xyxy)

    # class_predictions[class_id] 是一个列表，存储该类别的所有预测框信息
    # 每个预测框信息是一个字典：{'confidence': float, 'box': np.ndarray (4,), 'image_id': int}
    # box 坐标是原始图片像素坐标 [xmin, ymin, xmax, ymax]
    class_predictions = [[] for _ in range(num_classes)]  # 保存预测框（置信度、类别、框）

    # class_ground_truth[class_id] 是一个列表，存储该类别的所有真实框信息
    # 每个真实框信息是一个字典：{'box': np.ndarray (4,), 'image_id': int}
    # box 坐标是原始图片像素坐标 [xmin, ymin, xmax, ymax]
    class_ground_truth = [[] for _ in range(num_classes)]  # 保存真实框（类别、框）
    batch_idx = 0  # 记录当前处理的是第几个 Batch

    # 计算当前 epoch 在整个数据集中的起始图片索引，用于生成全局唯一的 image_id
    dataset_image_index_offset = (epoch - 1) * len(validation_dl.dataset)

    print(f"\n--- Starting Evaluation  for Epoch {epoch} ---")
    # 遍历验证集数据加载器
    # custom_collate_fn 返回的数据包括：
    # x_label: 图片张量 (B, C, H, W)
    # y_label: 真实标签张量 (B, Num_Anchors, num_classes + 4)，包含 One-Hot 类别和 padded_norm 边界框
    # objects_list: Batch 中每张图片的原始对象列表 (从 XML 解析得到) - **这个包含所有真实框！**
    # original_images: Batch 中每张图片的原始 numpy 数组 (H, W, C)
    # original_sizes: Batch 中每张图片的原始尺寸 (width, height) 列表
    # scales, pad_xs, pad_ys: Batch 中每张图片的预处理参数列表
    # padded_sizes: Batch 中每张图片的填充后尺寸 (padded_width, padded_height) 列表
    for x_label, y_label, objects_list, original_images, original_sizes, scales, pad_xs, pad_ys, padded_sizes in tqdm(
            validation_dl, desc=f"Evaluating Epoch {epoch}", total=len(validation_dl), ncols=100):
        x_label = x_label.to(device)
        # y_label can stay on CPU
        # anchor_boxes_normalized = validation_dl.dataset.reference_boxes.to(device)  # 形状 (Num_Anchors, 4)，归一化到填充后尺寸 [0, 1]
        # ✅ Anchor Boxes 已经通过参数传递进来，无需再次从 dataset 获取
        anchor_boxes_normalized = anchor_boxes_xyxy.to(device)

        batch_size, num_anchors, _ = y_label.shape
        with torch.no_grad():
            y_hat = model(
                x_label)  # 假设输出形状 (batch_size, num_anchors, num_classes+4)# 模型输出 (batch_size, Num_Anchors, num_classes+4)
            # 假设最后的 4 个值是原始边界框预测 [dx, dy, dw, dh]
            # 前 num_classes 个值是类别 Logits

            # --- Calculate validation loss for this batch ---
            # Need to ensure y_label is on the same device as y_hat for loss calculation
            y_label_device = y_label.to(y_hat.device)
            val_loss, val_cls_loss, val_box_loss = criterion(y_hat, y_label_device)
            total_val_loss += val_loss.item()
            total_val_cls_loss += val_cls_loss.item()  # Optional: accumulate component losses
            total_val_box_loss += val_box_loss.item()  # Optional: accumulate component losses

        # --- 后处理预测结果和处理真实框，用于计算 mAP 和可视化 ---
        # 将模型输出和真实标签从设备移动到 CPU 并转换为 numpy (用于后续的后处理和计算)
        y_hat_cpu = y_hat.detach().cpu()
        y_label_cpu = y_label.detach().cpu()  # 真实标签 (从 DataLoader 输出时通常已经在 CPU)
        # actual_cls_batch = y_label.cpu()[:, :, :num_classes] # 不再从 y_label 获取真实类别
        # actual_boxes_batch = y_label.cpu()[:, :, num_classes:] # 不再从 y_label 获取真实框
        pred_cls_logits_batch = y_hat.detach().cpu()[:, :, :num_classes]  # (B, Num_Anchors, num_classes)
        raw_pred_boxes_batch = y_hat.detach().cpu()[:, :, num_classes:]  # 原始预测 [dx, dy, dw, dh]

        # 计算类别概率 (使用 softmax)
        pred_conf_batch = F.softmax(pred_cls_logits_batch, dim=-1)
        # 获取每个 Anchor 预测的最高置信度及其对应的类别索引
        conf_of_predicted_class_batch, pred_cls_idx_batch = torch.max(pred_conf_batch,
                                                                      dim=2)  # 形状 (B, Num_Anchors), (B, Num_Anchors)

        # 初始化列表，用于收集第一个 Batch 的最终检测结果 (用于可视化)
        # 这个列表的结构是 List[List[Dict]]，Dict 包含 box, score, class_id
        # box 坐标是归一化到 padded_size [0, 1] 的
        final_detections_batch_for_vis = []

        # --- 逐图片处理当前 Batch 的预测结果和真实框 ---
        for i in range(batch_size):
            # 为当前图片生成唯一的 image_id (用于 mAP 计算)
            current_image_id = dataset_image_index_offset + (batch_idx * batch_size) + i

            # 获取当前图片对应的预处理参数 (缩放比例、填充量、原始尺寸、填充后尺寸)
            # 这些参数从 custom_collate_fn 返回的列表中获取
            orig_w, orig_h = original_sizes[i]
            scale_i = scales[i]
            pad_x_i = pad_xs[i]
            pad_y_i = pad_ys[i]
            padded_width_i, padded_height_i = padded_sizes[i]
            original_size_i = (orig_w, orig_h)
            padded_size_i = (padded_width_i, padded_height_i)  # Note: Should be padded_height_i

            # --- 将模型的原始预测框 [dx, dy, dw, dh] 解码为 padded_size 的 [xmin, ymin, xmax, ymax] ---
            # 使用 VisualizedPredict.py 中的 decode_boxes 函数
            # anchor_boxes_normalized 是 (Num_Anchors, 4) 在 device 上
            # raw_pred_boxes_batch[i] 是 (Num_Anchors, 4) 在 CPU 上
            # decode_boxes 期望输入在同一设备上，所以将 raw_pred_boxes_batch[i] 移到设备上
            decoded_boxes_i_padded_norm = decode_boxes(
                anchor_boxes_normalized,  # Anchor boxes (在 Device 上)
                raw_pred_boxes_batch[i].to(device)  # 当前图片的原始预测框 (移到 Device 上)
            ).cpu().numpy()  # 在 Device 上解码，然后移回 CPU 并转为 numpy

            # --- 将解码后的预测框从 padded_size 的归一化+坐标转换回原始图片像素坐标 ---
            # 这些坐标用于 NMS 的 IoU 计算以及 mAP 计算
            decoded_boxes_i_original_pixels = self.convert_padded_norm_to_original_pixel(
                decoded_boxes_i_padded_norm,  # 当前图片解码后的预测框 (Num_Anchors, 4) - padded_norm
                original_size=original_size_i,
                padded_size=padded_size_i,
                scale=scale_i,
                pad_x=pad_x_i,
                pad_y=pad_y_i
            )  # 形状 (Num_Anchors, 4) 在原始图片像素坐标系

            # --- 对预测框应用置信度阈值过滤和 NMS ---
            # 获取当前图片每个 Anchor 的预测置信度 (最高类别的概率) 和类别索引
            conf_of_predicted_class_i = conf_of_predicted_class_batch[i, :]  # 形状 (Num_Anchors,)
            pred_cls_idx_i = pred_cls_idx_batch[i, :]  # 形状 (Num_Anchors,)

            # 应用置信度阈值过滤初始预测框
            confidence_threshold = 0.05  # 定义您的置信度阈值 (例如 0.05 或更高)
            # 创建一个布尔掩码，标记出置信度高于阈值的 Anchor
            high_confidence_mask = (conf_of_predicted_class_i > confidence_threshold).numpy()  # NumPy 掩码

            # 使用掩码过滤边界框、置信度、类别索引
            # 过滤后的边界框使用原始像素坐标 (用于 NMS IoU)
            high_conf_boxes_original_pixels = decoded_boxes_i_original_pixels[
                high_confidence_mask]  # 形状 (Num_High_Conf_Anchors, 4)
            high_conf_scores = conf_of_predicted_class_i.numpy()[
                high_confidence_mask]  # 形状 (Num_High_Conf_Anchors,)
            high_conf_classes = pred_cls_idx_i.numpy()[high_confidence_mask]  # 形状 (Num_High_Conf_Anchors,)

            # 同时保留经过过滤后的预测框，但坐标是归一化到 padded_size 的，这些用于可视化
            high_conf_boxes_padded_norm = decoded_boxes_i_padded_norm[
                high_confidence_mask]  # 形状 (Num_High_Conf_Anchors, 4)

            # 对过滤后的预测框应用 NMS (逐类别进行)
            final_boxes_i_original_pixels = []  # 最终通过 NMS 的边界框 (原始像素坐标) - 用于 mAP
            final_boxes_i_padded_norm = []  # 最终通过 NMS 的边界框 (padded_norm 坐标) - 用于可视化
            final_scores_i = []  # 最终通过 NMS 的置信度
            final_classes_i = []  # 最终通过 NMS 的类别索引

            # 遍历所有前景类别
            for class_id in range(num_classes):
                if class_id == BACKGROUND_CLASS: continue  # 跳过背景类

                # 筛选出当前类别的预测结果
                class_mask = (high_conf_classes == class_id)
                if not np.any(class_mask):  # 如果当前类别没有高置信度预测框，跳过 NMS
                    continue

                # 获取当前类别的边界框和置信度，用于 NMS
                boxes_for_nms_original_pixels = high_conf_boxes_original_pixels[
                    class_mask]  # 当前类别的框 (原始像素坐标)
                boxes_for_nms_padded_norm = high_conf_boxes_padded_norm[class_mask]  # 当前类别的框 (padded_norm)
                scores_for_nms = scores_for_nms[class_mask]  # 当前类别的置信度 # Note: scores_for_nms is already filtered

                # 应用 NMS (使用原始像素坐标的框来计算 IoU)
                # apply_nms 返回的是在 boxes_for_nms 内部的索引
                keep_indices_in_class = self.apply_nms(boxes_for_nms_original_pixels, scores_for_nms,
                                                       iou_threshold=iou_threshold)  # 使用评估的 IoU 阈值

                # --- 存储通过 NMS 的最终检测结果 ---
                # 这些是当前图片经过解码、过滤和 NMS 后的最终检测结果
                # 存储原始像素坐标的框用于 mAP 计算
                final_boxes_i_original_pixels.extend(boxes_for_nms_original_pixels[keep_indices_in_class])
                # 存储 padded_norm 坐标的框用于可视化
                final_boxes_i_padded_norm.extend(boxes_for_nms_padded_norm[keep_indices_in_class])
                # 存储置信度和类别
                final_scores_i.extend(scores_for_nms[keep_indices_in_class])
                final_classes_i.extend([class_id] * len(keep_indices_in_class))  # 为每个保留的框添加类别 ID

            # --- 收集第一个 Batch 的最终检测结果，用于可视化 ---
            # 如果是第一个 Batch，将当前图片的最终检测结果格式化并添加到 batch_for_vis 列表中
            # 传递给 visualize_predictions 的最终检测框坐标是归一化到 padded_size [0, 1] 的
            if len(original_images) > 0 and batch_idx == 0:  # 只对第一个 Batch 进行可视化 (batch_idx 从 0 开始)
                # 准备可视化所需的 y_pre (包含第一个 Batch 所有解码后的预测框，归一化到 padded_size [0, 1])
                # Note: Re-calculating first batch data here for visualization.
                # In a production setting, you might handle this differently to avoid redundant computation.
                # For debugging and visualization of the first batch, this is acceptable.
                # Get the first item from the DataLoader iterator (if it's the first batch)
                try:
                    # To avoid iterating the DataLoader twice, we need to handle getting the first batch carefully.
                    # A simpler approach for visualization is to collect the decoded predictions for the *current* batch
                    # and pass the first image's data from the current batch.

                    # Collect decoded padded_norm boxes and class probabilities for the current batch (for visualization)
                    current_batch_decoded_padded_norm_boxes = []
                    current_batch_pred_cls_probs = []

                    # Iterate through images in the current batch to prepare data for visualization
                    for vis_img_idx in range(batch_size):
                        # Decode predicted boxes for visualization (padded_norm)
                        decoded_boxes_vis_img = decode_boxes(
                            anchor_boxes_normalized,  # Anchor boxes (on Device)
                            raw_pred_boxes_batch[vis_img_idx].to(device)  # Current image raw predictions (on Device)
                        ).cpu()  # Decode on device, move to CPU

                        current_batch_decoded_padded_norm_boxes.append(decoded_boxes_vis_img)
                        current_batch_pred_cls_probs.append(
                            pred_conf_batch[vis_img_idx].cpu())  # Class probabilities (on CPU)

                    # Stack for the current batch
                    current_batch_decoded_padded_norm_boxes_stacked = torch.stack(
                        current_batch_decoded_padded_norm_boxes, dim=0)  # (B, Num_Anchors, 4)
                    current_batch_pred_cls_probs_stacked = torch.stack(current_batch_pred_cls_probs,
                                                                       dim=0)  # (B, Num_Anchors, num_classes)

                    # Combine for visualization y_pre
                    y_pre_decoded_for_vis_batch = torch.cat(
                        (current_batch_pred_cls_probs_stacked, current_batch_decoded_padded_norm_boxes_stacked), dim=-1
                    )  # (B, Num_Anchors, num_classes + 4)

                    # Prepare final detections for visualization for the first image in the current batch
                    image_final_detections = []
                    # Find detections belonging to the first image (image_id = current_image_id - i + 0)
                    # Alternatively, collect final detections directly from the first image (i=0) loop above
                    # Let's refine the collection of final_detections_batch_for_vis

                    # Re-structure collecting final_detections_batch_for_vis to be more robust
                    # It should collect detections for all images in the first processed batch.
                    # Let's assume `final_boxes_i_padded_norm`, `final_scores_i`, `final_classes_i`
                    # from the loop over `i` contain the final detections for image `i` of the current batch.

                    # We need to collect these lists for the first batch outside the inner loop.
                    # This requires a small change before the inner loop:
                    # if batch_idx == 0:
                    #     first_batch_final_detections_per_image = []
                    # After the inner loop for image `i`:
                    # if batch_idx == 0:
                    #     image_final_detections = []
                    #     for j in range(len(final_boxes_i_padded_norm)):
                    #          image_final_detections.append({
                    #              'box': final_boxes_i_padded_norm[j],
                    #              'score': final_scores_i[j],
                    #              'class_id': final_classes_i[j]
                    #          })
                    #     first_batch_final_detections_per_image.append(image_final_detections)

                    # Then call visualize_predictions after the inner loop finishes for the first batch:
                    # if batch_idx == 0:
                    #      visualize_predictions(
                    #          original_images=original_images, # This is the list for the current batch
                    #          y_pre=y_pre_decoded_for_vis_batch.numpy(), # Decoded predictions for the current batch
                    #          y_batch=y_label_cpu.numpy(), # Ground truth for the current batch
                    #          output_dir=visualizations_dir,
                    #          epoch=epoch,
                    #          num_classes=num_classes,
                    #          scales=scales,
                    #          pad_xs=pad_xs,
                    #          pad_ys=pad_ys,
                    #          padded_sizes=padded_sizes,
                    #          final_detections=first_batch_final_detections_per_image # Final detections for the first batch
                    #      )

                    # Simplified approach for visualization: Just visualize the first image of the first batch
                    if batch_idx == 0:  # Only visualize the first batch
                        # Prepare data for the first image (i=0) in the current batch
                        first_image_original = original_images[0]
                        first_image_y_pre_decoded_for_vis = y_pre_decoded_for_vis_batch[0].unsqueeze(
                            0).numpy()  # (1, Num_Anchors, num_classes+4)
                        first_image_y_batch_cpu = y_label_cpu[0].unsqueeze(0).numpy()  # (1, Num_Anchors, num_classes+4)
                        first_image_scales = [scales[0]]
                        first_image_pad_xs = [pad_xs[0]]
                        first_image_pad_ys = [pad_ys[0]]
                        first_image_padded_sizes = [padded_sizes[0]]

                        # Prepare final detections for the first image (i=0) in the current batch
                        # These were collected into final_boxes_i_padded_norm etc. during the inner loop with i=0
                        first_image_final_detections = []
                        # Assuming final_boxes_i_padded_norm, final_scores_i, final_classes_i
                        # from the inner loop when i=0 are still accessible or re-collected
                        # For robustness, let's re-collect final detections for image 0
                        # This requires repeating part of the inner loop logic for i=0

                        # Re-collect high confidence and final detections for the first image (i=0)
                        conf_of_predicted_class_0 = conf_of_predicted_class_batch[0, :]
                        pred_cls_idx_0 = pred_cls_idx_batch[0, :]
                        decoded_boxes_0_original_pixels = decoded_boxes_i_original_pixels  # This variable holds the last calculated original pixel boxes (for image i)
                        decoded_boxes_0_padded_norm = decoded_boxes_i_padded_norm

                        high_confidence_mask_0 = (conf_of_predicted_class_0 > confidence_threshold).numpy()
                        high_conf_boxes_original_pixels_0 = decoded_boxes_0_original_pixels[high_confidence_mask_0]
                        high_conf_scores_0 = conf_of_predicted_class_0.numpy()[high_confidence_mask_0]
                        high_conf_classes_0 = pred_cls_idx_0.numpy()[high_confidence_mask_0]
                        high_conf_boxes_padded_norm_0 = decoded_boxes_0_padded_norm[high_confidence_mask_0]

                        final_boxes_0_original_pixels = []
                        final_boxes_0_padded_norm = []
                        final_scores_0 = []
                        final_classes_0 = []

                        for class_id in range(num_classes):
                            if class_id == BACKGROUND_CLASS: continue
                            class_mask_0 = (high_conf_classes_0 == class_id)
                            if not np.any(class_mask_0): continue

                            boxes_for_nms_original_pixels_0 = high_conf_boxes_original_pixels_0[class_mask_0]
                            boxes_for_nms_padded_norm_0 = high_conf_boxes_padded_norm_0[class_mask_0]
                            scores_for_nms_0 = high_conf_scores_0[class_mask_0]

                            keep_indices_in_class_0 = self.apply_nms(boxes_for_nms_original_pixels_0, scores_for_nms_0,
                                                                     iou_threshold=iou_threshold)

                            final_boxes_0_original_pixels.extend(
                                boxes_for_nms_original_pixels_0[keep_indices_in_class_0])
                            final_boxes_0_padded_norm.extend(boxes_for_nms_padded_norm_0[keep_indices_in_class_0])
                            final_scores_0.extend(scores_for_nms_0[keep_indices_in_class_0])
                            final_classes_0.extend([class_id] * len(keep_indices_in_class_0))

                        # Format final detections for the first image
                        for j in range(len(final_boxes_0_padded_norm)):
                            image_final_detections.append({
                                'box': final_boxes_0_padded_norm[j],  # 归一化到 padded_size [0, 1] 的 numpy 数组
                                'score': final_scores_0[j].item(),
                                'class_id': final_classes_0[j]
                            })
                        final_detections_batch_for_vis.append(
                            image_final_detections)  # Add detections for the first image

                        visualize_predictions(
                            original_images=[first_image_original],  # Pass as a list of one image
                            y_pre=first_image_y_pre_decoded_for_vis,
                            # Decoded predictions for the first image (padded_norm)
                            y_batch=first_image_y_batch_cpu,  # Ground truth for the first image (padded_norm)
                            output_dir=visualizations_dir,  # 可视化输出目录
                            epoch=epoch,  # 当前 epoch 数
                            num_classes=num_classes,
                            scales=first_image_scales,  # 缩放比例列表
                            pad_xs=first_image_pad_xs,  # X 填充量列表
                            pad_ys=first_image_pad_ys,  # Y 填充量列表
                            padded_sizes=first_image_padded_sizes,  # 填充后尺寸列表
                            final_detections=final_detections_batch_for_vis  # Final detections for the first image
                        )


            except Exception as e:
            print(f"Error during visualization of first batch, image {i}: {e}")
            import traceback
            traceback.print_exc()

    # --- 收集所有图片的预测结果和真实框，用于计算 mAP ---
    # 将经过 NMS 过滤后的预测结果 (使用原始像素坐标的框) 添加到 class_predictions 列表中
    for j in range(len(final_boxes_i_original_pixels)):
        class_id = final_classes_i[j]
        # 只收集前景类别的预测结果
        if class_id != BACKGROUND_CLASS:
            class_predictions[class_id].append({
                'confidence': final_scores_i[j].item(),
                'box': final_boxes_i_original_pixels[j],  # 存储原始像素坐标的框
                'image_id': current_image_id  # 图片的唯一 ID
            })

    # ✅ 将真实框添加到 class_ground_truth 列表中 (使用 objects_list)
    # objects_list[i] 是当前图片 i 的原始真实框列表
    current_image_original_gts = objects_list[i]

    for gt_obj in current_image_original_gts:
        # Assuming gt_obj is a dict like {'box': [x1, y1, x2, y2], 'class_id': class_id, ...}
        gt_box_original_pixel = gt_obj['box']
        gt_class_id = gt_obj['class_id']

        # Only add foreground ground truths for mAP calculation
        if gt_class_id != BACKGROUND_CLASS:
            # Optional: Check if the GT box is valid (e.g., meets a minimum size threshold)
            # width = gt_box_original_pixel[2] - gt_box_original_pixel[0]
            # height = gt_box_original_pixel[3] - gt_box_original_pixel[1]
            # min_box_size = 1 # Example minimum size
            # if width > min_box_size and height > min_box_size:
            class_ground_truth[gt_class_id].append({
                'box': np.array(gt_box_original_pixel),  # Ensure it's a numpy array
                'image_id': current_image_id
            })
        # else: Ignore small GT box

    # Batch 处理索引递增
    batch_idx += 1


# --- 计算 AP 和 mAP ---
# 遍历所有前景类别，计算 AP
# valid_classes 列表包含了所有前景类别的索引 (从 1 到 num_classes-1)
valid_classes = [i for i in range(num_classes) if i != BACKGROUND_CLASS]
aps = []  # 用于存储每个有效类别的 AP 值

print("\n--- Calculating AP per class ---")
for class_id in valid_classes:
    # 获取当前类别收集到的所有预测结果和真实框
    preds = class_predictions[class_id]  # 列表 of {'confidence', 'box', 'image_id'}
    gts = class_ground_truth[class_id]  # 列表 of {'box', 'image_id'}

    print(
        f"Class {class_id} (Name: {class_names.get(class_id, 'Unknown')}): {len(preds)} predictions, {len(gts)} ground truths")

    # 如果当前类别没有真实框，该类别的 AP 通常为 0
    if len(gts) == 0:
        ap = 0.0
        if len(preds) > 0:  # 有预测框但没有真实框
            print(f"Warning: Class {class_id} has {len(preds)} predictions but no ground truth instances.")
        else:  # 没有预测框也没有真实框
            print(f"Class {class_id} has no ground truth instances.")

    # 如果当前类别有真实框但没有预测框，AP 为 0
    elif len(preds) == 0:
        ap = 0.0
        print(f"Class {class_id} has {len(gts)} GTs but no predictions.")
    else:
        # --- 计算当前类别的 AP ---
        # 将预测结果按置信度降序排序
        all_preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        # 初始化 True Positive (TP) 和 False Positive (FP) 数组
        tp = np.zeros(len(all_preds_sorted))
        fp = np.zeros(len(all_preds_sorted))

        # 使用集合跟踪每个图片中已经被匹配过的真实框，避免重复匹配同一个真实框
        matched_gts_per_image = {}  # 字典：{image_id: set of gt_indices_in_gts_list}

        # 遍历按置信度排序后的预测结果
        for idx, pred in enumerate(all_preds_sorted):
            conf, pred_box, image_id = pred['confidence'], pred['box'], pred['image_id']

            best_iou = 0
            best_gt_idx_in_gts_list = -1  # 在当前类别真实框列表 'gts' 中的索引

            # 查找在同一图片中与当前预测框 IoU 最高且**尚未被匹配**的真实框
            for gt_idx_in_gts_list, gt_item in enumerate(gts):
                gt_box, gt_image_id = gt_item['box'], gt_item['image_id']

                if gt_image_id == image_id:  # 只考虑同一图片中的真实框
                    # 使用真实框在 'gts' 列表中的索引作为其在该类别的唯一 ID (在当前评估运行期间)
                    gt_unique_id = gt_idx_in_gts_list  # Use list index as unique ID within this class/image scope

                    # 检查该真实框是否已经在当前图片中被更高置信度的预测框匹配过
                    if image_id not in matched_gts_per_image:
                        matched_gts_per_image[image_id] = set()

                    if gt_unique_id not in matched_gts_per_image[image_id]:  # 只考虑尚未匹配的真实框
                        # 计算当前预测框与真实框的 IoU (框坐标已是原始像素坐标)
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou > best_iou:  # 找到最高的 IoU
                            best_iou = iou
                            best_gt_idx_in_gts_list = gt_idx_in_gts_list  # 记录最高 IoU 对应的真实框索引

            # --- 判断当前预测框是 TP 还是 FP ---
            if best_iou >= iou_threshold:
                # 如果最高 IoU 大于等于阈值，并且找到了一个未匹配的真实框
                if best_gt_idx_in_gts_list != -1:
                    # 是 True Positive (TP)
                    tp[idx] = 1
                    # 标记这个真实框在该图片中已经被匹配
                    matched_gts_per_image[image_id].add(best_gt_idx_in_gts_list)
                else:
                    # 如果最高 IoU 大于等于阈值，但找不到未匹配的真实框（说明重叠的真实框都已经被更高置信度的预测框匹配了）
                    # 这是 False Positive (FP)
                    fp[idx] = 1
            else:
                # 如果最高 IoU 小于阈值
                # 是 False Positive (FP)
                fp[idx] = 1

        # --- 计算 Precision 和 Recall ---
        # 累积 TP 和 FP
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        # Precision = 累积 TP / (累积 TP + 累积 FP)
        # 加上一个小的 epsilon 防止除以零
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)

        # Recall = 累积 TP / 当前类别的真实框总数
        total_gt_for_class = len(gts)
        recall = cumsum_tp / (total_gt_for_class + 1e-8)

        # --- 计算当前类别的 AP ---
        # 使用 VOC 2010 标准的 11 点插值方法
        try:
            # Ensure recall and precision are numpy arrays
            recall = np.array(recall)
            precision = np.array(precision)
            ap = self.calculate_AP(recall, precision)
            print(f"Class {class_id} AP: {ap:.4f}")
        except Exception as e:
            print(f"Error calculating AP for class {class_id}: {e}")
            ap = 0.0  # 如果计算出错，AP 设为 0

        # --- 绘制当前类别的 PR 曲线 ---
        # 只有在该类别有预测框和真实框时才绘制
        if len(preds) > 0 and total_gt_for_class > 0:
            try:
                plt.figure(figsize=(8, 6))
                # 绘制 Recall-Precision 曲线
                plt.plot(recall, precision,
                         label=f'{class_names.get(class_id, f"Class {class_id}")} (AP={ap:.4f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(
                    f'Class {class_id} PR Curve (Total GT: {total_gt_for_class}, Preds: {len(preds)})')
                plt.ylim(0.0, 1.05)  # 设置 Y 轴范围
                plt.xlim(0.0, 1.05)  # 设置 X 轴范围
                plt.legend(loc='lower left')  # 显示图例
                # 确保可视化输出目录存在
                if not os.path.exists(visualizations_dir):
                    os.makedirs(visualizations_dir)
                # 保存 PR 曲线图
                plt.savefig(
                    os.path.join(visualizations_dir, f"pr_curve_class_{class_id}_epoch_{epoch}.png"),
                    dpi=300,
                    bbox_inches='tight')
                plt.close()  # 关闭当前图表
            except Exception as e:
                print(f"绘制类别 {class_id} PR曲线失败: {str(e)}")

    # 将计算出的当前类别的 AP 添加到列表中
    aps.append(ap)

# --- 计算所有有效类别的 mAP ---
# 只对有真实框的类别计算 mAP (如果一个类别没有任何真实框，它的 AP 在一些评估标准下不计入 mAP)
# 这里我们已经将没有真实框的类别 AP 设为 0 了，所以直接对所有前景类别的 AP 求平均
# 确保 valid_classes 不为空
if len(valid_classes) > 0:
    mAP = sum(aps) / len(aps)
else:
    mAP = 0.0  # 如果没有前景类别，mAP 为 0

print(f"\nEpoch {epoch} mAP: {mAP:.4f}")

# --- 计算平均验证损失 ---
avg_val_total_loss = total_val_loss / len(validation_dl)
# 可选：计算并记录分量平均损失
# avg_val_cls_loss = total_val_cls_loss / len(validation_dl)
# avg_val_box_loss = total_val_box_loss / len(validation_dl)
print(f"Epoch {epoch} Validation Loss: {avg_val_total_loss:.4f}")

# 返回平均验证损失和 mAP
return avg_val_total_loss, mAP