import torch
#from mpmath.identification import transforms
import torchvision.transforms as transforms
#from pandas.conftest import axis_1
#from pyglet import model
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy, normalize
from torch.utils.data import DataLoader
from numpy import vstack, argmax
import numpy as np
from LossFunc import num_classes
from param import root_dir,TRAIN_BATCH_SIZE,VALIDATION_BATCH_SIZE,EPOCHS,LEARNING_RATE,Embeding_dim,Netdepth
import model
import Dataset
import LossFunc
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
class model_train(object):
    #def __init__(self,model):
    # 检查输入数据
    def check_input_data(x):
        print("Input data has NaN:", torch.isnan(x).any())
        print("Input data has inf:", torch.isinf(x).any())

    @staticmethod
    def compute_iou(boxA, boxB):
            # 计算两个边界框的交并比 (IoU)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

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
    def train(self,model):
        #将训练数据集分为8：1：1的训练集，验证集，数据集
        train_transform = transforms.Compose([
           # transforms.Resize((1600, 3040)),
            transforms.Resize((256, 256)),# 调整图片大小
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize(          # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
            ])

        val_transform=transforms.Compose([
            transforms.Resize((256,256)),
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

        train_dl=DataLoader(dataset=train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
        validation_dl=DataLoader(dataset=val_dataset,batch_size=VALIDATION_BATCH_SIZE,shuffle=True,drop_last=True)

    #使用adam优化
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

        # 初始化 GradScaler
        scaler = torch.amp.GradScaler("cuda")

        print('start training')
        with open('training_log.txt', 'w') as file:
            for epochs in range(EPOCHS):
                torch.autograd.set_detect_anomaly(True)
                for x_batch,y_batch in tqdm(train_dl, desc=f"Epoch {epochs+1}/{EPOCHS}", total=len(train_dl), ncols=100):
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
                        # print(f"y_pre tensor dtype: {y_pre.dtype}")#y_pre tensor dtype: torch.float16
                    assert y_pre.dtype is torch.float32
                    criterion = LossFunc.CustomLoss()  # ✅ 先实例化
                    loss = criterion(y_pre, y_batch)  # ✅ 正确调用
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
                    file.flush()
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()  # 缩放损失并反向传播

                    scaler.unscale_(optimizer)  # 取消缩放以便于梯度裁剪
                        # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新 GradScaler


                    #optimizer.step()#模型参数的更新体现在loss.backward()和optimizer.step()这两个步骤中。
#loss.backward()计算梯度，optimizer.step()应用这些梯度来更新模型的参数。
        #运用测试集计算更新模型的精确度
                test_accuracy=self.evaluate_model(validation_dl,model,num_classes);
                print("Epoch:",epochs+1,"loss:%.5f",loss.item(),"Accuracy:%.5f",test_accuracy)
                log_string = "Epoch: %d, loss: %.5f, Test accuracy: %.5f\n" % (epochs + 1, loss.item(), test_accuracy)

                if test_accuracy > best_accuracy + threshold:
                    best_accuracy = test_accuracy
                    no_improve_epochs = 0
                # Save the best model
                    torch.save(model, 'best_model.pth')
                    print("New best model saved with accuracy: {0:.5f}".format(best_accuracy))
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


    def evaluate_model(self, validation_dl, model, num_classes, iou_threshold=0.5, device='cuda'):
        predictions, actuals = [], []  # 初始化空列表
        per_image_detection = []  # 保存每个图像的预测和真实框信息
        class_predictions = [[] for _ in range(num_classes)]  # 保存预测框（置信度、类别、框）
        class_ground_truth = [[] for _ in range(num_classes)]  # 保存真实框（类别、框）

        for x_label, y_label in validation_dl:
            x_label, y_label = x_label.to(device), y_label.to(device)  # 将数据移动到指定设备
            with torch.no_grad():
                y_hat = model(x_label)  # 假设输出形状 (batch_size, num_anchors, num_classes+4)

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
                for j in range(num_anchors):
                    # 预测框
                    pred_box = pred_boxes[i, j].cpu().detach().numpy()
                    pred_confidence = pred_conf[i, j].item()
                    pred_class = pred_cls_idx[i, j].item()

                    # 真实框
                    actual_box = actual_boxes[i, j].cpu().detach().numpy()
                    actual_class = torch.argmax(actual_cls[i, j]).item()

                    # 保存预测框信息
                    class_predictions[pred_class].append({
                        'confidence': pred_confidence,
                        'box': pred_box,
                        'image_id': i  # 图像 ID，用于区分不同图像
                    })

                    # 保存真实框信息
                    class_ground_truth[actual_class].append({
                        'box': actual_box,
                        'image_id': i
                    })

        # 计算每个类别的 AP 和 mAP
        aps = [] #用于存储每个类别的 AP
        for class_id in range(num_classes):
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
                else:
                    fp[idx] = 1

            # 计算 Precision 和 Recall
            cumsum_tp = np.cumsum(tp)
            cumsum_fp = np.cumsum(fp)
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)
            recall = cumsum_tp / len(gts)

            # 计算 AP
            ap = self.calculate_AP(precision, recall)
            aps.append(ap)

        # 计算 mAP
        valid_aps = [ap for ap in aps if ap != -1]
        mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

        return mAP
if __name__=='__main__':
    print("build model")#imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
    #这里要写model.py的transformer函数，然后引import img_size=(1600,3040) ,
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_transformer=model.model(embed_dim=Embeding_dim,norm_layer=None,num_heads=4,hideF=256,
                     Pyin_channels=Embeding_dim,Pyout_channels=32,
                 num_classes=6,num_anchors=6,Netdepth=Netdepth) #Fimg看pyramid用例
    # 将模型转换为半精度 (FP16)
    #model_transformer=model_transformer.half()
    # 将模型移动到 GPU
    model_transformer = model_transformer.to(device)

    model_train().train(model_transformer)#model_train类的实例化
    torch.save(model_transformer,'model.pth')
