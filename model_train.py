import torch
from pandas.conftest import axis_1
from pyglet import model
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from numpy import vstack, argmax
from param import Test_File_Path,Train_File_Path,TRAIN_BATCH_SIZE,TEST_BATCH_SIZE,Validation_File_Path,VALIDATION_BATCH_SIZE,EPOCHS
from model import Transformer1
class model_train(object):
    def __init__(self):
        pass
    def train(self,model):
        #将训练数据集分为8：1：1的训练集，验证集，数据集
        train=CSVDATASET(Train_File_Path)
        validation=CSVDATASET(Validation_File_Path)   #通过方法，将验证集变为一个可以load的变量train
        train_dl=DataLoader(train,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
        validation_dl=DataLoader(validation,batch_size=VALIDATION_BATCH_SIZE,shuffle=True)

    #s使用adam优化
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        print('start training')
        for epochs in range(EPOCHS):
            for x_batch,y_batch in train_dl:
                y_batch=y_batch.long()
                optimizer.zero_grad()
                y_pre=model(x_batch)
                loss=cross_entropy(y_pre,y_batch)
                loss.backward()
                optimizer.step()#模型参数的更新体现在loss.backward()和optimizer.step()这两个步骤中。
#loss.backward()计算梯度，optimizer.step()应用这些梯度来更新模型的参数。
        #运用测试集计算更新模型的精确度
        accuracy=self.evaluate_model(validation_dl,model);
        print("Epoch:",epochs+1,"loss:%.5f",loss.item(),"Accuracy:%.5f",accuracy)

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
    print("build model")
    #这里要写model.py的transformer函数，然后引import
    model_transformer=Transformer1(nhead=10,             # number of heads in the multi-head-attention models
                           dim_feedforward=128,  # dimension of the feedforward network model in nn.TransformerEncoder
                           num_layers=1,
                           dropout=0.0,
                           classifier_dropout=0.3)
    model1=model_train(model)#model_train类的实例化
    torch.save(model1,'model.pth')
