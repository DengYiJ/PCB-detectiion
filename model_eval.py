from importlib.resources import contents
import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor
import torch.nn.functional as F
model=torch.load('model.pth')
label_dict, char_dict = load_file_file()#加载标签字典，字符字典，写在feature.py里面
label_dict_inv=[]#补充，创造反向label，用于最后得到预测分类
print(label_dict)

def eval(text)
    labels=['汽车']
    contents=[text]
    #写一个函数，将文本转化为模型输入特征，返回值为输入特征，标签
    input_feature,input_label=text_feature(labels,contents,label_dict,char_dict)
    input_feature=tensor(input_feature)#将特征转换为张量
    label_pred=model(input_feature)
    label_pred=F.softmax(label_pred,dim=1)#将预测结果转换为概率分布
    label_pred=label_pred.cpu().detach().numpy()#一个张量（Tensor）转换为NumPy数组，同时确保这个数组不会与原始的计算图相关联
    predict_list = np.argmax(label_pred, axis=1).tolist()
    return label_dict_inv[predict_list[0]]

