import torch
#from mpmath.identification import transforms
import torchvision.transforms as transforms
#from pandas.conftest import axis_1
from pyglet import model
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
        scaler = GradScaler("cuda")

        print('start training')
        with open('training_log.txt', 'w') as file:
            for epochs in range(EPOCHS):
                torch.autograd.set_detect_anomaly(True)
                for x_batch,y_batch in tqdm(train_dl, desc=f"Epoch {epochs+1}/{EPOCHS}", total=len(train_dl), ncols=100):
                    print(f"x_batch type: {type(x_batch)}, shape: {x_batch.shape if isinstance(x_batch, torch.Tensor) else 'Not a tensor'}")
                # x_batch: 图像张量 (batch_size, channels, height, width)
                # y_batch: 标签张量 (batch_size, num_anchors, num_classes + 4)前 `num_classes` 列是分类标签（one-hot 编码），后 4 列是边界框标签。
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_batch=y_batch.long()
                    print(f"y_batch type: {type(y_batch)}, shape: {y_batch.shape}")
                    optimizer.zero_grad()
                    # 使用混合精度训练
                    with autocast(dtype=torch.float16):  #自动混合精度上下文
                        y_pre=model(x_batch)#`y_pre`：模型输出，形状为 `(batch_size, num_anchors, num_classes + 4)`。前 `num_classes` 列是分类预测。 后 4 列是边界框预测（`[x, y, w, h]`）。
                        assert y_pre.dtype is torch.float16
                        criterion = LossFunc.CustomLoss()  # ✅ 先实例化
                        loss = criterion(y_pre, y_batch)  # ✅ 正确调用
                        #print(f"loss dtype: {loss.dtype}")
                        #assert loss.dtype is torch.float16
                    # loss.backward()
                    scaler.scale(loss).backward()  # 缩放损失并反向传播
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
    def evaluate_model(self,validation_dl,model,num_classes,iou_threshold=0.5):
        predictions,actuals=[],[]#初始化空列表
        per_image_detection = []  # 保存每个图像的预测和真实框信息
        class_predictions = [[] for _ in range(num_classes)]  # 保存预测框（置信度、类别、框）
        class_ground_truth = [[] for _ in range(num_classes)]  # 保存真实框（类别、框）
       # num_classes=num_classes
        for x_label,y_label in validation_dl:
            y_hat=model(x_label) # 假设输出形状 (batch_size, num_anchors, num_classes+4)
            y_hat=y_hat.detach().numpy()#yhat.detach().numpy() 这一步是为了确保在评估或预测时不干扰模型的训练过程，同时允许你将张量数据转换为更通用的格式
            actual_label=y_label.detach().numpy()
            #将预测标签和真实标签转换为类标签
            # 提取分类标签部分并取 argmax
            # 提取分类标签和边界框部分
            actual_cls = actual_label[:, :, :num_classes]  # (batch_size, num_anchors, num_classes)
            actual_boxes = actual_label[:, :, num_classes:]  # (batch_size, num_anchors, 4)
            pred_cls = y_hat[:, :, :num_classes]  # (batch_size, num_anchors, num_classes)
            pred_boxes = y_hat[:, :, num_classes:]  # (batch_size, num_anchors, 4)
            # 置信度（每个锚点的最大分类概率）
            pred_conf = np.max(pred_cls, axis=2)  # 置信度（每个锚点的最大分类概率）
            pred_cls_idx = np.argmax(pred_cls, axis=2)  # 预测类别

            batch_size = pred_boxes.shape[0]
            num_anchors = pred_boxes.shape[1]
            for i in range(batch_size):
                for j in range(num_anchors):
                    # 预测框
                    pred_box = pred_boxes[i, j]
                    pred_confidence = pred_conf[i, j]
                    pred_class = pred_cls_idx[i, j]

                    # 真实框
                    actual_box = actual_boxes[i, j]
                    actual_class = np.argmax(actual_cls[i, j])

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
                    aps = []
                    for class_id in range(num_classes):
                        # 获取当前类别的预测和真实框
                        preds = class_predictions[class_id]
                        gts = class_ground_truth[class_id]

                        if len(gts) == 0:
                            aps.append(-1)  # 如果没有真实框，AP 设置为 -1 表示无效
                            continue

                        # 统计所有真实框
                        all_gts = []
                        for gt in gts:
                            all_gts.append(gt['box'])

                        # 统计所有预测框
                        all_preds = []
                        for pred in preds:
                            all_preds.append({
                                'confidence': pred['confidence'],
                                'box': pred['box'],
                                'image_id': pred['image_id']
                            })

                        # 排序预测框（按置信度降序）
                        all_preds_sorted = sorted(all_preds, key=lambda x: x['confidence'], reverse=True)

                        # 初始化 TP 和 FP
                        tp = np.zeros(len(all_preds_sorted))
                        fp = np.zeros(len(all_preds_sorted))

                        # 遍历每个预测框
                        for idx, pred in enumerate(all_preds_sorted):
                            matched = False
                            # 遍历当前图像的真实框
                            for gt in gts:
                                if gt['image_id'] == pred['image_id']:
                                    iou = compute_iou(pred['box'], gt['box'])
                                    if iou >= iou_threshold:
                                        matched = True
                                        # 防止重复匹配
                                        gts.pop(gts.index(gt))
                                        break
                            if matched:
                                tp[idx] = 1
                            else:
                                fp[idx] = 1

                        # 计算 Precision 和 Recall
                        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp) + 1e-8)
                        recall = np.cumsum(tp) / len(gts)

                        # 计算 AP
                        ap = calculate_AP(precision, recall)
                        aps.append(ap)

                    # 计算 mAP
                    valid_aps = [ap for ap in aps if ap != -1]
                    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

                    return mAP

def calculate_AP(precision, recall):
    """计算 AP"""
    m = np.concatenate(([0], recall, [1]))
    p = np.concatenate(([0], precision, [0]))
    for i in range(len(p) - 1, 0, -1):
        p[i - 1] = max(p[i - 1], p[i])
    indices = np.where(m[1:] != m[:-1])[0]
    AP = np.sum((m[indices] - m[indices - 1]) * p[indices])
    return AP

def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou
if __name__=='__main__':
    print("build model")#imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
    #这里要写model.py的transformer函数，然后引import img_size=(1600,3040) ,
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_transformer=model.model(embed_dim=Embeding_dim,norm_layer=None,num_heads=4,hideF=256,
                     Pyin_channels=Embeding_dim,Pyout_channels=32,
                 num_classes=6,num_anchors=6,Netdepth=Netdepth) #Fimg看pyramid用例
    model_train().train(model_transformer)#model_train类的实例化
    torch.save(model_transformer,'model.pth')
