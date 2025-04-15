# -*- coding: utf-8 -*-
import math
import torch
#from mpmath.identification import transforms
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
#from pandas.conftest import axis_1
#from pyglet import model
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from torch.nn.functional import cross_entropy, normalize
from torch.utils.data import DataLoader
from numpy import vstack, argmax
import numpy as np
from LossFunc import num_classes
from param import root_dir,TRAIN_BATCH_SIZE,VALIDATION_BATCH_SIZE,EPOCHS,LEARNING_RATE,Embeding_dim,Netdepth,visualizations_dir
import model
import Dataset
import LossFunc
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from model1 import model1
from VisualizedPredict import visualize_predictions
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
   # 应用 NMS - 使用torchvision.ops.nms
from torchvision.ops import nms
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class model_train(object):
    #def __init__(self,model):
    # 检查输入数据
    def check_input_data(x):
        print("Input data has NaN:", torch.isnan(x).any())
        print("Input data has inf:", torch.isinf(x).any())

    @staticmethod
    def compute_iou(boxA, boxB):
        
        if any(math.isnan(x) for x in boxA) or any(math.isnan(x) for x in boxB):
            print(f"检测到非法NaN值：boxA={boxA}, boxB={boxB}")
            return 0.0
        # 计算两个边界框的交并比 (IoU)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA +1) * max(0, yB - yA+1 )

        boxAArea = (boxA[2] - boxA[0]+1 ) * (boxA[3] - boxA[1]+1)
        boxBArea = (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1 )

        # iou = interArea / float(boxAArea + boxBArea - interArea+ 1e-8)

        return interArea / float(boxAArea + boxBArea - interArea+ 1e-8)

    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        # 将边界框和分数转换为 PyTorch 张量
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # 确保张量在CPU上
        boxes_tensor = boxes_tensor.cpu()
        scores_tensor = scores_tensor.cpu()
        # 应用 NMS
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)

        # 返回保留的边界框和分数
        nms_boxes = boxes_tensor[keep_indices].numpy()
        nms_scores = scores_tensor[keep_indices].numpy()

        return nms_boxes, nms_scores

    @staticmethod
    def calculate_AP(precision, recall):
        # 使用VOC2010标准计算平均精度 (AP)
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.

        return ap

    @staticmethod    # 添加自定义的 collate_fn
    def custom_collate_fn(batch):
        # 分离不同类型的数据
        images = [item[0] for item in batch]  # 变换后的图像
        labels = [item[1] for item in batch]  # 标签
        original_images = [item[2] for item in batch]  # 原始图像
        original_sizes = [item[3] for item in batch]  # 原始尺寸

        # # 找到当前批次中的最大尺寸
        # max_h = max([img.shape[1] for img in images])
        # max_w = max([img.shape[2] for img in images])
        # 对图像进行填充
        padded_images = []
        for img in images:
            # 计算需要填充的量
            pad_h = 1024 - img.shape[1]
            pad_w = 1024 - img.shape[2]

        # 对称填充（左、右、上、下）
        padded_img = F.pad(img, (pad_w//2, pad_w - pad_w//2, 
                                pad_h//2, pad_h - pad_h//2), 
                         value=0)
        padded_images.append(padded_img)
        # 堆叠变换后的图像和标签（这些应该是相同尺寸的）
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        # 原始图像和尺寸保持为列表形式
        return images, labels, original_images, original_sizes

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

        # train_dl=DataLoader(dataset=train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
        # validation_dl = DataLoader(dataset=val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, drop_last=True)
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
    #使用adam优化
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

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
                for x_batch,y_batch,original_images, original_sizes in tqdm(train_dl, desc=f"Epoch {epochs+1}/{EPOCHS}", total=len(train_dl), ncols=100):
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
                    criterion = LossFunc.CustomLoss()  # ✅ 先实例化
                    loss,l_cls, l_box = criterion(y_pre, y_batch)  # ✅ 正确调用
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
                    file.write(f"Epoch {epochs + 1}")
                    file.write(f"x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
                    file.write(f"y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")
                    file.write(f"y_pre shape: {y_pre.shape}, dtype: {y_pre.dtype}")
                    file.write(f"y_batch:{y_batch.detach().cpu().numpy()}")
                    file.write(f"y_pre:{y_pre.detach().cpu().numpy()}")
                    file.write(f"Loss: {loss.item()}")
                    # file.flush()
                   # with torch.autograd.detect_anomaly():
                    scaler.scale(loss).backward()  # 缩放损失并反向传播

                    scaler.unscale_(optimizer)  # 取消缩放以便于梯度裁剪
                        # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新 GradScaler
                    # 打印每个epoch的平均损失
                    avg_total = total_loss / len(train_dl)
                    avg_cls = cls_loss / len(train_dl)
                    avg_box = box_loss / len(train_dl)

                    file.write(
                        f"Epoch {epochs + 1}: Total Loss={avg_total:.4f}, Class Loss={avg_cls:.4f}, Box Loss={avg_box:.4f}")
                    file.flush()
                    # # 在训练循环中直接使用 y_pre 进行可视化
                    # visualize_predictions( original_images, y_pre.detach().cpu(), y_batch.cpu(), visualizations_dir,
                    #                       epochs + 1, num_classes=6, original_sizes=original_sizes )

                    #optimizer.step()#模型参数的更新体现在loss.backward()和optimizer.step()这两个步骤中。
#loss.backward()计算梯度，optimizer.step()应用这些梯度来更新模型的参数。
        #运用测试集计算更新模型的精确度
                MAP=self.evaluate_model(validation_dl,model,num_classes,epochs+1);
                print("Epoch:",epochs+1,"loss:%.5f",loss.item(),"MAP:%.5f",MAP)
                log_string = "Epoch: %d, loss: %.5f,MAP: %.5f\n" % (epochs + 1, loss.item(), MAP)

                if MAP > best_accuracy + threshold:
                    best_accuracy = MAP
                    no_improve_epochs = 0
                # Save the best model
                    torch.save(model, 'best_model.pth')
                    print("New best model saved with MAP: {0:.5f}".format(best_accuracy))
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print("Early stopping triggered.")
                        break
                #清空显存
                torch.cuda.empty_cache()
                scheduler.step()
            # 打印到控制台
            # print(log_string.strip())
            # 写入到文件
                file.write(log_string)

            # 训练完成后，再次清空显存
            torch.cuda.empty_cache()

    #是怎么评估模型好坏的呢？用验证集！


    def evaluate_model(self, validation_dl, model, num_classes,epoch, iou_threshold=0.5, device='cuda'):
        # model.eval()
        predictions, actuals = [], []  # 初始化空列表
        per_image_detection = []  # 保存每个图像的预测和真实框信息
        class_predictions = [[] for _ in range(num_classes)]  # 保存预测框（置信度、类别、框）
        class_ground_truth = [[] for _ in range(num_classes)]  # 保存真实框（类别、框）
        batch_idx = 0
        for x_label, y_label, original_images, original_sizes in validation_dl:
            x_label, y_label = x_label.to(device), y_label.to(device)  # 将数据移动到指定设备
            with torch.no_grad():
                y_hat = model(x_label)  # 假设输出形状 (batch_size, num_anchors, num_classes+4)
                # 对分类预测应用softmax
               # cls_pred = F.softmax(cls_pred, dim=-1)
                # 在每个epoch的第一个batch进行可视化
                if batch_idx == 0:
                    visualize_predictions(
                        original_images,
                        y_hat.detach().cpu(),
                        y_label.cpu(),
                        visualizations_dir,
                        epoch,  # 现在可以正确传递 epoch
                        num_classes=7,
                        original_sizes=original_sizes
                    )
                batch_idx += 1

            # 提取分类标签部分并取 argmax
                actual_cls = y_label[:, :, :num_classes]  # (batch_size, num_anchors, num_classes)
                actual_boxes = y_label[:, :, num_classes:]  # (batch_size, num_anchors, 4)
                pred_cls = y_hat[:, :, :num_classes]  # (batch_size, num_anchors, num_classes)
                pred_boxes = y_hat[:, :, num_classes:]  # (batch_size, num_anchors, 4)

            # 置信度（每个锚点的最大分类概率）
                pred_conf, pred_cls_idx = torch.max(pred_cls, dim=2)  # (batch_size, num_anchors)
                '''torch.max(pred_cls, dim=2)：在分类预测的最后一个维度（类别维度）上取最大值。
pred_conf：每个锚点的最大分类概率（置信度），形状为 (batch_size, num_anchors)。
pred_cls_idx：每个锚点的预测类别索引，形状为 (batch_size, num_anchors)。'''
                batch_size = pred_boxes.shape[0]
                num_anchors = pred_boxes.shape[1]

            for i in range(batch_size):
                 # 新增NMS处理逻辑
    # 提取当前图像的所有预测框和置信度
                image_boxes = pred_boxes[i]  # [num_anchors, 4]
                image_scores = pred_conf[i]  # [num_anchors]
    
                # 应用NMS (阈值设为0.5)
                nms_boxes, nms_scores = self.apply_nms(image_boxes.cpu().numpy(), 
                                         image_scores.cpu().numpy(),
                                         iou_threshold=0.5)
                for j in range(len(nms_boxes)):
                    # 预测框
                    # pred_box = pred_boxes[i, j].cpu().detach().numpy()
                    # pred_confidence = pred_conf[i, j].item()
                    # pred_class = pred_cls_idx[i, j].item()
                    pred_box = nms_boxes[j]
                    pred_confidence = nms_scores[j]
                    class_idx = np.argmax(pred_cls[i, j].cpu().numpy())

                    # 保存预测框信息
                    class_predictions[class_idx].append({
                        'confidence': pred_confidence,
                        'box': pred_box,
                        'image_id': i  # 图像 ID，用于区分不同图像
                    })
                     # 处理真实框（保持原样）
                for j in range(num_anchors):
                     # 真实框
                    actual_box = actual_boxes[i, j].cpu().detach().numpy()
                    actual_class = torch.argmax(actual_cls[i, j]).item()

                    # 保存真实框信息
                    class_ground_truth[actual_class].append({
                        'box': actual_box,
                        'image_id': i
                    })

        # 计算每个类别的 AP 和 mAP
        valid_classes = [i for i in range(1, num_classes)]  # 跳过class 0（背景）
        aps = [] #用于存储每个类别的 AP
        all_pred_labels = []
        all_true_labels = []
        for class_id in valid_classes:
            # 获取当前类别的预测和真实框
            preds = class_predictions[class_id]
            gts = class_ground_truth[class_id]

            if len(gts) == 0:
                aps.append(-1)  # 如果没有真实框，AP 设置为 -1 表示无效
                continue

            # 统计所有真实框
            all_gts = [(gt['box'], gt['image_id']) for gt in gts]

            # 统计所有预测框
            all_preds = [(pred['confidence'], pred['box'], pred['image_id']) for pred in preds]

            # 排序预测框（按置信度降序）
            all_preds_sorted = sorted(all_preds, key=lambda x: x[0], reverse=True)

            # 初始化 TP 和 FP
            tp = np.zeros(len(all_preds_sorted))
            fp = np.zeros(len(all_preds_sorted))

            # 标记已经匹配的真实框
            matched_gts = set()

            # 遍历每个预测框
            for idx, (conf, pred_box, image_id) in enumerate(all_preds_sorted):
                matched = False
                # 遍历当前图像的真实框
                for k, (gt_box, gt_image_id) in enumerate(all_gts):
                    if gt_image_id == image_id and k not in matched_gts:
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            matched = True
                            # 防止重复匹配
                            matched_gts.add(k)
                            break
                if matched:
                    tp[idx] = 1
                    all_pred_labels.append(class_id)
                    all_true_labels.append(class_id)
                else:
                    fp[idx] = 1
                    all_pred_labels.append(class_id)
                    all_true_labels.append(-1)  # 表示负样本


            # 计算 Precision 和 Recall
            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)
            recall = cumsum_tp / len(gts)
             # 添加对单点或空预测的处理
            if len(precision) == 0 or len(recall) == 0:
                print(f"Class {class_id} 完全无预测数据")
                ap = 0.0
                valid_samples = 0
            elif len(precision) < 2 or len(recall) < 2:
                print(f"Class {class_id} 有效数据点不足（仅{len(precision)}个）")
                ap = 0.0
                valid_samples = 0
            else:
                ap = self.calculate_AP(precision, recall)
                valid_samples = len(gts)

            # 统一绘图逻辑（增加多维检查）
            if valid_samples > 0 and len(precision) >= 2 and len(recall) >= 2 and ap > 0:
                try:
                    # 添加输入校验
                    assert len(recall) == len(precision), "Recall与Precision长度不一致"
                    
                    plt.figure(figsize=(8, 6))
                    auc_value = auc(recall, precision)
                    plt.plot(recall, precision, 
                            label=f'Class {class_id} (AUC={auc_value:.2f}, AP={ap:.2f})')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Class {class_id} PR曲线 (有效样本:{valid_samples})')
                    plt.legend(loc='lower left')
                    plt.savefig(f"./image/pr_curve_class_{class_id}.png", dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"绘制类别 {class_id} PR曲线失败: {str(e)}")
                    ap = 0.0  # 失败时重置AP

            aps.append(ap)  # 确保所有情况都记录AP值


        # 计算 mAP
        valid_aps = [ap for ap in aps if ap != -1]
        mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

        # 计算混淆矩阵
        label_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=valid_classes)
        self.plot_confusion_matrix(cm, label_names)
        print(f"正在评估类别 {class_id}，真实框数量：{len(gts)}")  # 应显示1-6类
        return mAP

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


if __name__=='__main__':
    print("build model")#imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
    #这里要写model.py的transformer函数，然后引import img_size=(1600,3040) ,
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_transformer=model.model(embed_dim=Embeding_dim,norm_layer=None,num_heads=4,hideF=256,
                     Pyin_channels=Embeding_dim,Pyout_channels=32,
                 num_classes=7,num_anchors=6,Netdepth=Netdepth) #Fimg看pyramid用例
    # 将模型转换为半精度 (FP16)
    #model_transformer=model_transformer.half()
    # 将模型移动到 GPU
    model_transformer = model_transformer.to(device)

    model_train().train(model_transformer)#model_train类的实例化
    torch.save(model_transformer,'model.pth')
