import os
import cv2
import numpy as np
import torch

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

def visualize_predictions(original_images, y_pre, y_batch, output_dir, epoch, num_classes,scales, pad_xs, pad_ys, padded_sizes):
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
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(original_images)):
        # 复制原始图片以便在其上绘制
        # original_images 是 numpy 数组列表 (H x W x C)
        image = original_images[i].copy()
        orig_height, orig_width = image.shape[:2]
        print(f"orig_height:{orig_height}, orig_width:{orig_width}")
        # 计算resize时的缩放比例和padding
        scale_i = scales[i]
        pad_x_i = pad_xs[i]
        pad_y_i = pad_ys[i]
        padded_width_i, padded_height_i = padded_sizes[i]
        # padded_size_i = (padded_width_i, padded_height_i)
        # 绘制预测框
        for anchor in range(y_pre[i].shape[0]):
            # print(f"start drawing anchor {anchor}。")
            # y_pre[i] 是当前图片的预测结果 (NumPy 数组)
            # 切片结果是 NumPy 数组切
            class_probs_pred = y_pre[i, anchor, :num_classes] # 预测的类别概率 (包含背景)
            bbox_raw_pred = y_pre[i, anchor, num_classes:] # 预测的边界框 ( assumed 0-1 in padded_size)
            class_idx_pred = np.argmax(class_probs_pred) # 预测的类别索引
            score_pred = class_probs_pred[class_idx_pred] # 预测类别的置信度
            print(f"class_probs_pred: {class_probs_pred}。")
            print(f"bbox_raw_pred: {bbox_raw_pred}。")
            print(f"class_idx_pred: {class_idx_pred}。")
            print(f"score_pred: {score_pred}。")

            if class_idx_pred == 0 or score_pred<=0.01:# 过滤掉背景类预测和低置信度预测
                continue

            if score_pred > 0.01:  # 0.5
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
            gt_color = (0, 0, 255)  # 红色 (BGR 格式)

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
        print(f"Image saved to {output_path}")

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