import torch
#from mpmath.identification import transforms
import torchvision.transforms as transforms
#from pandas.conftest import axis_1
from pyglet import model
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy, normalize
from torch.utils.data import DataLoader
from numpy import vstack, argmax
from param import root_dir,TRAIN_BATCH_SIZE,VALIDATION_BATCH_SIZE,EPOCHS,LEARNING_RATE
import model
import Dataset
import LossFunc
from torch.optim.lr_scheduler import StepLR
class model_train(object):
    #def __init__(self,model):

    def train(self,model):
        #将训练数据集分为8：1：1的训练集，验证集，数据集
        train_transform = transforms.Compose([
            transforms.Resize((1600, 3040)),  # 调整图片大小
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize(          # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
            ])

        val_transform=transforms.Compose([
            transforms.Resize((1600,3040)),
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

        train_dl=DataLoader(dataset=train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
        validation_dl=DataLoader(dataset=val_dataset,batch_size=VALIDATION_BATCH_SIZE,shuffle=True)

    #使用adam优化
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        print('start training')
        with open('training_log.txt', 'w') as file:
            for epochs in range(EPOCHS):
                for x_batch,y_batch in train_dl:
                    print(f"x_batch type: {type(x_batch)}, shape: {x_batch.shape if isinstance(x_batch, torch.Tensor) else 'Not a tensor'}")
                # x_batch: 图像张量 (batch_size, channels, height, width)
                # y_batch: 标签张量 (batch_size, num_anchors, num_classes + 4)前 `num_classes` 列是分类标签（one-hot 编码），后 4 列是边界框标签。
                    y_batch=y_batch.long()
                    optimizer.zero_grad()
                    y_pre=model(x_batch)#`y_pre`：模型输出，形状为 `(batch_size, num_anchors, num_classes + 4)`。前 `num_classes` 列是分类预测。 后 4 列是边界框预测（`[x, y, w, h]`）。
                    loss=LossFunc.CustomLoss.forward(y_pre,y_batch)
                    loss.backward()
                    optimizer.step()#模型参数的更新体现在loss.backward()和optimizer.step()这两个步骤中。
#loss.backward()计算梯度，optimizer.step()应用这些梯度来更新模型的参数。
        #运用测试集计算更新模型的精确度
                test_accuracy=self.evaluate_model(validation_dl,model);
                print("Epoch:",epochs+1,"loss:%.5f",loss.item(),"Accuracy:%.5f",accuracy)
                log_string = "Epoch: %d, loss: %.5f, Test accuracy: %.5f\n" % (epoch + 1, loss.item(), test_accuracy)

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
            scheduler.step()
            # 打印到控制台
            # print(log_string.strip())
            # 写入到文件
            file.write(log_string)

    #是怎么评估模型好坏的呢？用验证集！
    def evaluate_model(self,validation_dl,model):
        predictions,actuals=[],[]#初始化空列表
        for x_label,y_label in validation_dl:
            y_hat=model(x_label)
            y_hat=y_hat.detach().numpy()#yhat.detach().numpy() 这一步是为了确保在评估或预测时不干扰模型的训练过程，同时允许你将张量数据转换为更通用的格式
            actual_label=y_label.detach().numpy()
            #将预测标签和真实标签转换为类标签
            actual_label=argmax(actual_label,axis=1)
            actual_label=actual_label.reshape(len(actual_label),1)
            y_hat=y_hat.reshape(len(y_hat),1)#转换为向量，reshape是torch中自带的函数，不改变内容，只改变张量形式
            predictions.append(y_hat)
            actuals.append(actual_label)
            predictions,actuals=vstack(predictions),vstack(actuals)
            acc = accuracy_score(actuals, predictions)
            return acc

if __name__=='__main__':
    print("build model")#imghimgw是1600x3040，inchannel是3，patchsize是160，embeddim是768
    #这里要写model.py的transformer函数，然后引import img_size=(1600,3040) ,
    model_transformer=model.model(patch_size=160,in_channels=3,embed_dim=768,norm_layer=None,num_heads=4,hideF=256,
                 imgH=1600,imgW=3040,Pyin_channels=768,Pyout_channels=256,
                 FimgH=4,FimgW=4,num_classes=4,num_anchors=6) #Fimg看pyramid用例
    model_train().train(model_transformer)#model_train类的实例化
    torch.save(model_transformer,'model.pth')
