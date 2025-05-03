# -*- coding: utf-8 -*-
import math
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from torch.utils.data import DataLoader
import numpy as np
# ✅ 从 LossFunc 中导入 num_classes 或直接在 param 中定义并导入，确保一致性
# from LossFunc import num_classes # 如果 num_classes 在 LossFunc 中定义
# 或者从 param 导入 num_classes 如果在 param 中定义
# from param import num_classes # 确保 param.py 中有 num_classes 定义
# 这里假设 num_classes 在 param 中定义，或者您手动设置，确保与您的类别数一致
num_classes = 7 # <--- 确保这里的类别数与您的实际类别数一致，包含背景类

from param import root_dir, TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, EPOCHS, LEARNING_RATE, Embeding_dim, Netdepth, \
    visualizations_dir, Num_Anchors
import model
import Dataset
import LossFunc
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from VisualizedPredict import visualize_predictions,decode_boxes,class_names,visualize_predictions_new
import torch.nn.functional as F
   # 应用 NMS - 使用torchvision.ops.nms
from torchvision.ops import nms
import os
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
BACKGROUND_CLASS = 0

# 确保 visualizations_dir 存在
os.makedirs(visualizations_dir, exist_ok=True)

class model_train(object):
    def __init__(self):
        # Initialize lists to store training and validation losses and mAP
        self.train_losses = []
        self.eval_losses = [] # Store average validation loss per epoch
        self.eval_maps = [] # Store validation mAP per epoch
        # Ensure visualizations_dir is accessible or passed
        # self.visualizations_dir = visualizations_dir # Assuming visualizations_dir is imported from param

    def check_input_data(x):
        print("Input data has NaN:", torch.isnan(x).any())
        print("Input data has inf:", torch.isinf(x).any())

    @staticmethod
    def compute_iou(boxA, boxB):

        if any(math.isnan(x) for x in boxA) or any(math.isnan(x) for x in boxB):
            print(f"检测到非法NaN值：boxA={boxA}, boxB={boxB}")
            return 0.0
        # Ensure x1 < x2 and y1 < y2 for both boxes
        boxA = np.array([min(boxA[0], boxA[2]), min(boxA[1], boxA[3]), max(boxA[0], boxA[2]), max(boxA[1], boxA[3])])
        boxB = np.array([min(boxB[0], boxB[2]), min(boxB[1], boxB[3]), max(boxB[0], boxB[2]), max(boxB[1], boxB[3])])
        # 计算两个边界框的交并比 (IoU)

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # interArea = max(0, xB - xA +1) * max(0, yB - yA+1 )
        interArea = max(0, xB - xA ) * max(0, yB - yA )

        # boxAArea = (boxA[2] - boxA[0]+1 ) * (boxA[3] - boxA[1]+1)
        # boxBArea = (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1 )
        boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1] )
        # iou = interArea / float(boxAArea + boxBArea - interArea+ 1e-8)

        return interArea / float(boxAArea + boxBArea - interArea+ 1e-8)

    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        # boxes: numpy array (N, 4) in (x1, y1, x2, y2) format
        # scores: numpy array (N,)
        # iou_threshold: float
        # 将边界框和分数转换为 PyTorch 张量
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # 确保张量在CPU上
        boxes_tensor = boxes_tensor.cpu()
        scores_tensor = scores_tensor.cpu()
        # 应用 NMS
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)

        # 返回保留的边界框和分数
        # nms_boxes = boxes_tensor[keep_indices].numpy()
        # nms_scores = scores_tensor[keep_indices].numpy()

        # return nms_boxes, nms_scores
        # 返回保留的原始索引 (相对于输入的 boxes 和 scores 数组)
        return keep_indices.numpy()  # Return numpy array of indices

    @staticmethod
    def calculate_AP(precision, recall):
        # 使用VOC2010标准计算平均精度 (AP)
        # 从后向前遍历，对 precision 进行插值
        for i in range(len(precision) - 2, -1, -1):
             precision[i] = np.maximum(precision[i], precision[i + 1])
        # 计算 11 个点的平均精度
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            # 找到 recall >= t 的最大 precision
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.

        return ap


    @staticmethod
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


    @staticmethod    # 添加自定义的 collate_fn
    def custom_collate_fn(batch):
        # batch is a list of tuples: [(transformed_image, y_batch, objects, original_image_np, original_size, scale, pad_x, pad_y, padded_size), ...]

        # Separate the data types using correct indices
        images = [item[0] for item in batch]  # transformed_image
        labels = [item[1] for item in batch]  # y_batch
        objects_list = [item[2] for item in batch]  # Collect the list of objects for each image
        original_images_np = [item[3] for item in batch]  # original_image_np
        original_sizes = [item[4] for item in batch]  # (original_img_width, original_img_height) - This is what we need!
        scales = [item[5] for item in batch]  # scale
        pad_xs = [item[6] for item in batch]  # pad_x
        pad_ys = [item[7] for item in batch]  # pad_y
        padded_sizes = [item[8] for item in batch]  # padded_size

        # Stack the tensors (images and labels)
        images_batch = torch.stack(images, 0)
        labels_batch = torch.stack(labels, 0)

        # Return stacked tensors and lists of other data
        return images_batch, labels_batch,objects_list,original_images_np, original_sizes, scales, pad_xs, pad_ys, padded_sizes

    def train(self,model):
        #将训练数据集分为8：1：1的训练集，验证集，数据集
        train_transform = transforms.Compose([
           # transforms.Resize((1024, 1024)),
           #  transforms.Resize((256, 256)),# 调整图片大小
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize(          # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
            ])

        val_transform=transforms.Compose([
            # transforms.Resize((1024,1024)),
           # transforms.RandomVerticleFlip(),
            transforms.ToTensor(),
            transforms.Normalize(  # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )] )#(Validation_File_Path)
        # 通过方法，将验证集变为一个可以load的变量train
        best_accuracy = 0.0  # 用于跟踪最佳准确度
        best_model = None  # 用于存储最佳模型
        no_improve_epochs = 0
        threshold = 0.0001
        patience = 70
        train_dataset= Dataset.MyDataset(root_dir, transform=train_transform,train=True)
        val_dataset= Dataset.MyDataset(root_dir, transform=val_transform,train=False)
        # print("各类别样本统计:")
        # for i in range(num_classes):count = sum(1 for _, label, _, _ in train_dataset if torch.argmax(label[0, :num_classes]) == i)
        # print(f"Class {i}: {count} samples")

        train_dl = DataLoader(
            dataset=train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn  # 添加自定义的 collate_fn
        )

        validation_dl = DataLoader(
            dataset=val_dataset,
            batch_size=VALIDATION_BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn  # 验证集也需要使用相同的 collate_fn
        )

        try:
            # 从训练集的 dataset 获取 Anchor Boxes (假设它们是 PyTorch 张量或可以转换为张量)
            # 确保 Anchor Boxes 已经被移动到与模型相同的设备上
            anchor_boxes_xyxy = train_dl.dataset.reference_boxes.to(device)
            print(f"Successfully loaded anchor boxes with shape: {anchor_boxes_xyxy.shape}")
        except AttributeError:
            print("Error: train_dl.dataset does not have 'reference_boxes' attribute.")
            print("Please ensure your Dataset calculates and stores reference_boxes.")
            # 根据你的实际情况，可能需要在这里退出或采取其他错误处理
            exit()  # 或者 raise Exception(...)

        #使用adam优化
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

        criterion=LossFunc.CustomLoss(num_classes=num_classes, anchor_boxes_xyxy=anchor_boxes_xyxy)
        criterion.to(device)

        # 初始化 GradScaler
        scaler = torch.amp.GradScaler("cuda")

        print('start training')
        with open('training_log.txt', 'w') as file:
            for epochs in range(EPOCHS):
                # torch.autograd.set_detect_anomaly(True)
                total_loss = 0
                cls_loss = 0
                box_loss = 0

                model.train()
                for x_batch, y_batch, objects_list, original_images, original_sizes, scales, pad_xs, pad_ys, padded_sizes in tqdm(train_dl, desc=f"Epoch {epochs+1}/{EPOCHS}", total=len(train_dl), ncols=100):
                    # print(f"x_batch type: {type(x_batch)}, shape: {x_batch.shape if isinstance(x_batch, torch.Tensor) else 'Not a tensor'}")
                # x_batch: 图像张量 (batch_size, channels, height, width)
                # y_batch: 标签张量 (batch_size, num_anchors, num_classes + 4)前 `num_classes` 列是分类标签（one-hot 编码），后 4 列是边界框标签。
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    assert not torch.isnan(x_batch).any(), "x_batch contains NaN"
                    assert not torch.isinf(x_batch).any(), "x_batch contains Inf"
                    assert not torch.isnan(y_batch).any(), "y_batch contains NaN"
                    assert not torch.isinf(y_batch).any(), "y_batch contains Inf"
                    #如果检测到 NaN 或 Inf，assert 会抛出一个 AssertionError，并附带相应的错误信息（如 "x_batch contains NaN"）。
                    # 强制将输入数据转换为 FP16
                   # x_batch = x_batch.half()
                    if y_batch.dtype != torch.float32:
                        y_batch = y_batch.float()
                    # print(f"y_batch type: {type(y_batch)}, shape: {y_batch.shape}")
                    optimizer.zero_grad()
                    # 使用混合精度训练
                    #with torch.amp.autocast(device_type="cuda",dtype=torch.float16):  #自动混合精度上下文
                    y_pre=model(x_batch)#`y_pre`：模型输出，形状为 `(batch_size, num_anchors, num_classes + 4)`。前 `num_classes` 列是分类预测。 后 4 列是边界框预测（`[x, y, w, h]`）。
                    # print(f"y_pre tensor dtype: {y_pre.dtype},“y_pre shape:{y_pre.shape}")#y_pre tensor dtype: torch.float16
                    # assert y_pre.dtype is torch.float32

                    # ✅ 在这里获取 Anchor Boxes 并传递给 CustomLoss
                    # anchor_boxes_xyxy = train_dl.dataset.reference_boxes.to(device) # 从训练集的 dataset 获取 Anchor Boxes
                    # criterion =criterion(num_classes=num_classes, anchor_boxes_xyxy=anchor_boxes_xyxy)  # ✅ 实例化 LossFunc
                    loss,l_cls, l_box = criterion(y_pre, y_batch)  # ✅ 计算损失
                    # 记录损失
                    total_loss += loss.item()
                    cls_loss += l_cls.item()
                    box_loss += l_box.item()
                    # print("Loss requires_grad:", loss.requires_grad)
                        # print(f"loss dtype: {loss.dtype}") #FP32
                        #assert loss.dtype is torch.float16
                        # 打印中间输出以检查NaN或Inf值
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN or Inf detected in loss: {loss}")
                        break
                    # loss.backward()
                    # file.write(f"Epoch {epochs + 1}")
                    # file.write(f"x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
                    # file.write(f"y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")
                    # file.write(f"y_pre shape: {y_pre.shape}, dtype: {y_pre.dtype}")
                    # file.write(f"y_batch:{y_batch.detach().cpu().numpy()}")
                    # file.write(f"y_pre:{y_pre.detach().cpu().numpy()}")
                    # file.write(f"Loss: {loss.item()}")
                    # file.flush()
                   # with torch.autograd.detect_anomaly():
                    scaler.scale(loss).backward()  # 缩放损失并反向传播

                    scaler.unscale_(optimizer)  # 取消缩放以便于梯度裁剪
                        # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新 GradScaler
                    # 打印每个epoch的平均损失
                avg_train_total_loss = total_loss / len(train_dl)
                self.train_losses.append(avg_train_total_loss)
                # avg_cls = cls_loss / len(train_dl)
                # avg_box = box_loss / len(train_dl)

                # ✅ 在评估时传递 Anchor Boxes 给 evaluate_model
                avg_val_loss,mAP=self.evaluate_model(validation_dl,model,num_classes,epochs+1,anchor_boxes_xyxy=val_dataset.reference_boxes.to(device))

                # Record evaluation loss and mAP
                self.eval_losses.append(avg_val_loss)
                self.eval_maps.append(mAP)
                print(f"Epoch: {epochs+1}, Train Loss: {avg_train_total_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val mAP: {mAP:.4f}")
                log_string = f"Epoch: {epochs + 1}, Train Loss: {avg_train_total_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val mAP: {mAP:.4f}\n"


                if mAP > best_accuracy + threshold:
                    best_accuracy = mAP
                    no_improve_epochs = 0
                # Save the best model
                    torch.save(model, 'best_model.pth')
                    print("New best model saved with MAP: {0:.5f}".format(best_accuracy))
                else:
                    no_improve_epochs += 1
                    print(f"Early stopping counter: {no_improve_epochs}/{patience}")
                    if no_improve_epochs >= patience:
                        print("Early stopping triggered.")
                        break
                #清空显存
                scheduler.step()
                # Write log to file
                file.write(log_string)
                file.flush()  # Ensure data is written immediately
                # Clear GPU cache (optional but can help with memory)
                torch.cuda.empty_cache()
            # After training loop finishes, plot the loss and mAP curves
            self.plot_loss_curves()  # Call the new plotting method
        # 训练完成后，再次清空显存
        torch.cuda.empty_cache()

    #是怎么评估模型好坏的呢？用验证集！

    # ✅ 修改 evaluate_model 方法签名，接收 anchor_boxes_xyxy 参数
    def evaluate_model(self, validation_dl, model, num_classes, epoch, anchor_boxes_xyxy, iou_threshold=0.5, device='cuda'):
        model.eval() # Set model to evaluation mode
        # Initialize variables for validation loss calculation
        total_val_loss = 0
        total_val_cls_loss = 0 # Optional: log component losses
        total_val_box_loss = 0 # Optional: log component losses
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
        batch_idx = 0# 记录当前处理的是第几个 Batch

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
        for x_label, y_label, objects_list, original_images, original_sizes,scales, pad_xs, pad_ys, padded_sizes in tqdm(validation_dl, desc=f"Evaluating Epoch {epoch}", total=len(validation_dl), ncols=100):
            x_label = x_label.to(device)
            # y_label can stay on CPU
            # anchor_boxes_normalized 是 (Num_Anchors, 4)，归一化到填充后尺寸 [0, 1]，已通过参数传入并移动到 device
            anchor_boxes_normalized = anchor_boxes_xyxy # ✅ 直接使用传入的参数

            batch_size, num_anchors, _ = y_label.shape
            with torch.no_grad():
                y_hat = model(x_label)  # 假设输出形状 (batch_size, num_anchors, num_classes+4)# 模型输出 (batch_size, Num_Anchors, num_classes+4)
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
            y_label_cpu = y_label.detach().cpu()  # 真实标签 (从 DataLoader 输出时通常已经在 CPU)  xywh

            pred_cls_logits_batch = y_hat_cpu[:, :, :num_classes]# (B, Num_Anchors, num_classes)
            raw_pred_boxes_batch = y_hat_cpu[:, :, num_classes:] # 原始预测 [dx, dy, dw, dh]

            # 计算类别概率 (使用 softmax)
            pred_conf_batch = F.softmax(pred_cls_logits_batch, dim=-1)
            # 获取每个 Anchor 预测的最高置信度及其对应的类别索引
            conf_of_predicted_class_batch, pred_cls_idx_batch = torch.max(pred_conf_batch,dim=2)  # 形状 (B, Num_Anchors), (B, Num_Anchors)

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
                padded_size_i = (padded_width_i, padded_height_i)

                # --- 将模型的原始预测框 [dx, dy, dw, dh] 解码为 padded_size 的 [xmin, ymin, xmax, ymax] ---
                # 使用 VisualizedPredict.py 中的 decode_boxes 函数
                # anchor_boxes_normalized 是 (Num_Anchors, 4) 在 device 上 (因为已经传入并 .to(device) 了)
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
                confidence_threshold = 0.7  # 定义您的置信度阈值 (例如 0.05 或更高)
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
                high_conf_boxes_padded_norm = decoded_boxes_i_padded_norm[high_confidence_mask]  # 形状 (Num_High_Conf_Anchors, 4)

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
                    scores_for_nms_class = high_conf_scores[class_mask]  # 当前类别的置信度

                    # 应用 NMS (使用原始像素坐标的框来计算 IoU)
                    # apply_nms 返回的是在 boxes_for_nms 内部的索引
                    keep_indices_in_class = self.apply_nms(boxes_for_nms_original_pixels, scores_for_nms_class,
                                                           iou_threshold=iou_threshold)  # 使用评估的 IoU 阈值

                    # --- 存储通过 NMS 的最终检测结果 ---
                    # 这些是当前图片经过解码、过滤和 NMS 后的最终检测结果
                    # 存储原始像素坐标的框用于 mAP 计算
                    final_boxes_i_original_pixels.extend(boxes_for_nms_original_pixels[keep_indices_in_class])
                    # 存储 padded_norm 坐标的框用于可视化
                    final_boxes_i_padded_norm.extend(boxes_for_nms_padded_norm[keep_indices_in_class])
                    # 存储置信度和类别
                    final_scores_i.extend(scores_for_nms_class[keep_indices_in_class])
                    final_classes_i.extend([class_id] * len(keep_indices_in_class))  # 为每个保留的框添加类别 ID
                    current_image_id = f"{i * validation_dl.batch_size + batch_idx}"  # <--- 为当前图片生成一个唯一的 ID
                    # print(f"Image {current_image_id} after NMS: len(final_scores_i) = {len(final_scores_i)}")
                    # print(f"  Scores: {final_scores_i[:5]}")
                    # print(f"  Classes: {final_classes_i[:5]}")
                    #把结果储存在    class_predictions用于map计算
                for k in range(len(final_scores_i)):
                    pred_box = final_boxes_i_original_pixels[k]  # 使用原始像素坐标的框用于 mAP
                    score = final_scores_i[k].item()  # 转换为标量
                    predicted_class_id = final_classes_i[k]  # 已经是整数类别 ID

                    class_predictions[predicted_class_id].append({
                            'confidence': score,
                            'box': pred_box.tolist() if isinstance(pred_box, torch.Tensor) else list(pred_box),
                            # 确保是列表或 numpy 数组，取决于 final_boxes_i_original_pixels 的类型
                            'image_id': current_image_id
                        })
                    # print(f"  Appended prediction for class {predicted_class_id}, confidence {score:.2f}. Current class_predictions[{predicted_class_id}] length: {len(class_predictions[predicted_class_id])}")
                # ✅ 将真实框添加到 class_ground_truth 列表中 (使用 objects_list)
                # objects_list[i] 是当前图片 i 的原始真实框列表 (从 XML 解析得到)
                current_image_original_gts = objects_list[i]

                for gt_obj in current_image_original_gts:
                     # Assuming gt_obj is a dict like {'box': [x1, y1, x2, y2], 'class_id': class_id, ...}
                     gt_box_original_pixel = gt_obj['bbox']
                     gt_class_id = gt_obj['class_idx'] # ✅ 使用 'class_idx' key 获取整数类别 ID

                     # Only add foreground ground truths for mAP calculation
                     # Ensure gt_class_id is not None and not background class
                     if gt_class_id is not None and gt_class_id != BACKGROUND_CLASS:
                         # Optional: Check if the GT box is valid (e.g., meets a minimum size threshold)
                         # width = gt_box_original_pixel[2] - gt_box_original_pixel[0]
                         # height = gt_box_original_pixel[3] - gt_box_original_pixel[1]
                         # min_box_size = 1 # Example minimum size
                         # if width > min_box_size and height > min_box_size:
                        class_ground_truth[gt_class_id].append({
                                 'box': np.array(gt_box_original_pixel),  # Ensure it's a numpy array
                                 'image_id': current_image_id
                             })
                        # print(f"  Appended gt for class {gt_class_id}. length: {len(class_ground_truth[gt_class_id])}")
                         # else: Ignore small or invalid GT box


            # Batch 处理索引递增
            batch_idx += 1

            # --- 对第一个 Batch 进行可视化 ---
            # 在处理完第一个 Batch 后，调用 visualize_predictions 函数
            if batch_idx == 1 and len(original_images) > 0: # ✅ 在第一个 Batch 处理完后且有图片数据时调用一次
                try:
                    # 准备可视化所需的 y_pre (包含第一个 Batch 所有解码后的预测框，归一化到 padded_size [0, 1])
                    # 需要对第一个 Batch 的原始预测框进行解码
                    # 使用 batch_idx == 0 时的原始数据 (如果内存允许) 或者重新从 DataLoader 获取第一个 Batch (效率低)
                    # 最好的方法是在上面的 Batch 循环中，当 batch_idx == 0 时，缓存所需的张量。
                    print("\n--- Preparing data for visualization of the first batch ---")
                    # 创建一个新的 DataLoader 迭代器，只获取第一个 Batch
                    # 注意：这种方式会额外加载一次数据，可能影响性能。
                    # 更好的方法是在上面的 Batch 循环中，当 batch_idx == 0 时，缓存所需的张量。
                    first_batch_iterator = iter(validation_dl)
                    (first_batch_x_label_vis, first_batch_y_label_vis,
                     first_batch_objects_list_vis, first_batch_original_images_vis,
                     first_batch_original_sizes_vis, first_batch_scales_vis,
                     first_batch_pad_xs_vis, first_batch_pad_ys_vis,
                     first_batch_padded_sizes_vis) = next(first_batch_iterator)

                    # 将图片张量移动到设备进行模型推理
                    first_batch_x_label_vis = first_batch_x_label_vis.to(device)

                    with torch.no_grad():
                         first_batch_y_hat_vis = model(first_batch_x_label_vis) # 获取第一个 Batch 的模型预测结果

                    # 将预测结果移回 CPU 并提取 raw predictions 和 class probabilities
                    first_batch_y_hat_cpu_vis = first_batch_y_hat_vis.detach().cpu()
                    # 原始预测 [dx, dy, dw, dh]，形状 (Batch_Size, Num_Anchors, 4)
                    first_batch_raw_pred_boxes_vis = first_batch_y_hat_cpu_vis[:, :, num_classes:]
                    # 类别 Logits，形状 (Batch_Size, Num_Anchors, num_classes)
                    first_batch_pred_cls_logits_vis = first_batch_y_hat_cpu_vis[:, :, :num_classes]
                    # 类别概率，形状 (Batch_Size, Num_Anchors, num_classes)
                    first_batch_pred_cls_probs_vis = F.softmax(first_batch_pred_cls_logits_vis, dim=-1)


                    # ✅ 修改这里：展平预测框和 Anchor Boxes，然后调用 decode_boxes
                    batch_size_vis, num_anchors_vis, _ = first_batch_raw_pred_boxes_vis.shape

                    # 展平原始预测框为 (Batch_Size * Num_Anchors, 4)
                    raw_pred_boxes_flat_vis = first_batch_raw_pred_boxes_vis.view(-1, 4).to(device)

                    # 为整个 Batch 重复 Anchor Boxes 并展平为 (Batch_Size * Num_Anchors, 4)
                    anchor_boxes_flat_vis = anchor_boxes_normalized.unsqueeze(0).repeat(batch_size_vis, 1, 1).view(-1, 4)


                    # 解码展平后的预测框到 padded normalized [0, 1]，结果形状 (Batch_Size * Num_Anchors, 4)
                    decoded_boxes_padded_norm_flat_vis = decode_boxes(
                        anchor_boxes_flat_vis, # 展平的 Anchor boxes (在 Device 上)
                        raw_pred_boxes_flat_vis # 展平的原始预测框 (在 Device 上)
                    ).cpu() # 解码后移回 CPU

                    # 将解码后的预测框重新 reshape 回 (Batch_Size, Num_Anchors, 4)
                    decoded_boxes_padded_norm_for_vis_batch = decoded_boxes_padded_norm_flat_vis.view(batch_size_vis, num_anchors_vis, 4)


                    # 真实标签 y_batch 也需要 padded_norm 坐标，并且在 CPU 上  xywh?
                    first_batch_y_label_cpu_vis = first_batch_y_label_vis.detach().cpu()


                    # 拼接类别概率和解码后的边界框，形成用于可视化的 y_pre (NumPy 数组)
                    y_pre_decoded_for_vis = torch.cat(
                        (first_batch_pred_cls_probs_vis, decoded_boxes_padded_norm_for_vis_batch), dim=-1
                    ).numpy()

                    visualize_predictions_new(
                        original_images=first_batch_original_images_vis,
                        y_pre=first_batch_y_hat_vis,  # <-- 注意这里应该传入模型的原始输出 y_pre,而不是解码后的结果
                        y_batch=first_batch_y_label_vis,  # <-- 注意这里应该传入原始的 y_batch (gt dx, dy, dw, dh)
                        anchor_boxes_xyxy=anchor_boxes_xyxy,  # <--- 传入 Anchor Boxes 张量
                        output_dir=visualizations_dir,
                        epoch=epoch,
                        num_classes=num_classes,
                        scales=first_batch_scales_vis,
                        pad_xs=first_batch_pad_xs_vis,
                        pad_ys=first_batch_pad_ys_vis,
                        padded_sizes=first_batch_padded_sizes_vis,
                        # 新函数内部会处理置信度过滤、NMS 和最终检测结果的生成
                        # final_detections 参数在新函数中不需要作为输入传递
                    )
                    print(f"Visualized first batch of Epoch {epoch}.")

                except Exception as e:
                    print(f"Error during visualization of first batch in Epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()


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

            # 如果当前类别没有真实框，该类别的 AP 通常为 0 (在某些评估标准下可能会忽略)
            if len(gts) == 0:
                ap = 0.0
                if len(preds) > 0: # 有预测框但没有真实框
                     print(f"Warning: Class {class_id} has {len(preds)} predictions but no ground truth instances.")
                else: # 没有预测框也没有真实框
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
                    # Note: Iterating through all gts for this class for every prediction is inefficient for large datasets.
                    # Consider optimizing this part for performance if needed (e.g., using spatial indexing or pre-calculating IoUs).
                    # For now, this simple loop is correct for mAP logic.
                    for gt_idx_in_gts_list, gt_item in enumerate(gts):
                        gt_box, gt_image_id = gt_item['box'], gt_item['image_id']

                        if gt_image_id == image_id:  # 只考虑同一图片中的真实框
                            # Use the index within the 'gts' list as a temporary unique ID for this GT during evaluation
                            gt_unique_id = gt_idx_in_gts_list

                            # Check if this ground truth box has already been matched by a higher confidence prediction in this image
                            if image_id not in matched_gts_per_image:
                                matched_gts_per_image[image_id] = set()

                            if gt_unique_id not in matched_gts_per_image[image_id]:  # Only consider unmatched ground truths
                                # Calculate IoU between the prediction box and the ground truth box (both in original pixel coordinates)
                                iou = self.compute_iou(pred_box, gt_box)
                                if iou > best_iou:  # Find the highest IoU
                                    best_iou = iou
                                    best_gt_idx_in_gts_list = gt_idx_in_gts_list  # Record the index of the best matching ground truth

                    # --- Determine if the current prediction is a True Positive (TP) or False Positive (FP) ---
                    if best_iou >= iou_threshold:
                        # If the highest IoU is greater than or equal to the threshold, and we found an unmatched ground truth
                        if best_gt_idx_in_gts_list != -1:
                            # It's a True Positive (TP)
                            tp[idx] = 1
                            # Mark this ground truth box as matched in this image
                            matched_gts_per_image[image_id].add(best_gt_idx_in_gts_list)
                        else:
                            # If the highest IoU is >= threshold, but no unmatched ground truth was found
                            # (meaning all overlapping ground truths have been matched by higher confidence predictions)
                            # It's a False Positive (FP)
                            fp[idx] = 1
                    else:
                        # If the highest IoU is less than the threshold
                        # It's a False Positive (FP)
                        fp[idx] = 1

                # --- Calculate Precision and Recall ---
                # Cumulate TP and FP
                cumsum_tp = np.cumsum(tp)
                cumsum_fp = np.cumsum(fp)

                # Precision = Cumulated TP / (Cumulated TP + Cumulated FP)
                # Add a small epsilon to prevent division by zero
                precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)

                # Recall = Cumulated TP / Total number of ground truths for this class
                total_gt_for_class = len(gts)
                recall = cumsum_tp / (total_gt_for_class + 1e-8)

                # --- Calculate AP for the current class ---
                # Use the VOC 2010 standard 11-point interpolation method
                try:
                    # Ensure recall and precision are numpy arrays
                    recall = np.array(recall)
                    precision = np.array(precision)
                    ap = self.calculate_AP(recall, precision)
                    print(f"Class {class_id} AP: {ap:.4f}")
                except Exception as e:
                    print(f"Error calculating AP for class {class_id}: {e}")
                    ap = 0.0  # Set AP to 0 if calculation fails

                # --- Plot PR Curve for the current class ---
                # Only plot if there are predictions and ground truths for this class
                if len(preds) > 0 and total_gt_for_class > 0:
                    try:
                        plt.figure(figsize=(8, 6))
                        # Plot Recall-Precision curve
                        plt.plot(recall, precision,
                                 label=f'{class_names.get(class_id, f"Class {class_id}")} (AP={ap:.4f})')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title(
                            f'Class {class_id} PR Curve (Total GT: {total_gt_for_class}, Preds: {len(preds)})')
                        plt.ylim(0.0, 1.05)  # Set Y-axis limits
                        plt.xlim(0.0, 1.05)  # Set X-axis limits
                        plt.legend(loc='lower left')  # Show legend
                        # Ensure visualizations_dir exists
                        if not os.path.exists(visualizations_dir):
                            os.makedirs(visualizations_dir)
                        # Save the PR curve plot
                        plt.savefig(
                            os.path.join(visualizations_dir, f"pr_curve_class_{class_id}_epoch_{epoch}.png"),
                            dpi=300,
                            bbox_inches='tight')
                        plt.close()  # Close the current plot
                    except Exception as e:
                        print(f"绘制类别 {class_id} PR曲线失败: {str(e)}")

            # 将计算出的当前类别的 AP 添加到列表中
            aps.append(ap)

        # --- Calculate mAP across all valid classes ---
        # Average the APs of all foreground classes
        # Ensure there is at least one valid class to avoid division by zero
        aps_to_average = [aps[i] for i, class_id in enumerate(valid_classes)]
        mAP = sum(aps_to_average) / len(aps_to_average) if len(aps_to_average) > 0 else 0.0
        print(f"\nEpoch {epoch} mAP: {mAP:.4f}")

        # --- Calculate average validation loss ---
        avg_val_total_loss = total_val_loss / len(validation_dl)
        # Optional: calculate and log component average losses
        # avg_val_cls_loss = total_val_cls_loss / len(validation_dl)
        # avg_val_box_loss = total_val_box_loss / len(validation_dl)
        print(f"Epoch {epoch} Validation Loss: {avg_val_total_loss:.4f}")

        # Return average validation loss and mAP
        return avg_val_total_loss, mAP

    def plot_confusion_matrix(self, cm, label_names, title='Confusion matrix', cmap=plt.cm.Reds):
        plt.close('all')
        plt.figure(figsize=(10, 8))
        plt.matshow(cm, cmap=cmap)  # 根据最下面的图按自己需求更改颜色

        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

        num_local = np.array(range(len(label_names)))
        plt.xticks(num_local, label_names, rotation=90)  # 将标签印在x轴坐标上
        plt.yticks(num_local, label_names)  # 将标签印在y轴坐标上
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig("./image/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    # --- New method to plot loss and mAP curves ---
    def plot_loss_curves(self):
        """Plots the training and validation loss curves and validation mAP curve."""
        # Ensure visualizations_dir exists
        if not os.path.exists(visualizations_dir):
            os.makedirs(visualizations_dir)

        epochs_range = range(1, len(self.train_losses) + 1)

        # Plot Loss Curves
        if self.train_losses or self.eval_losses: # Check if lists are not empty
            plt.figure(figsize=(10, 6))
            if self.train_losses:
                 plt.plot(epochs_range, self.train_losses, label='Train Loss')
            if self.eval_losses:
                 plt.plot(epochs_range, self.eval_losses, label='Validation Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss per Epoch')
            plt.legend()
            plt.grid(True)
            loss_curve_path = os.path.join(visualizations_dir, "loss_curve.png")
            plt.savefig(loss_curve_path, dpi=300)
            plt.close()
        else:
            print("No loss data to plot.")


        # Plot mAP Curve
        if self.eval_maps: # Check if list is not empty
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range, self.eval_maps, label='Validation mAP', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Validation mAP per Epoch')
            plt.legend()
            plt.grid(True)
            mAP_curve_path = os.path.join(visualizations_dir, "map_curve.png")
            plt.savefig(mAP_curve_path, dpi=300)
            plt.close()
        else:
            print("No mAP data to plot.")

if __name__=='__main__':
    print("build model")#imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
    #这里要写model.py的transformer函数，然后引import img_size=(1600,3040) ,
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_transformer=model.model(embed_dim=Embeding_dim,norm_layer=None,num_heads=4,hideF=256,
                     Pyin_channels=Embeding_dim,Pyout_channels=256,
                 num_classes=num_classes) # ✅ 使用导入的 num_classes 或手动设置的 num_classes

    # 将模型移动到 GPU
    model_transformer = model_transformer.to(device)

    model_train().train(model_transformer)#model_train类的实例化

    # ✅ 训练完成后保存模型
    # torch.save(model_transformer, 'model.pth') # 如果需要保存最终模型（通常保存最佳模型就够了）