# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:35
# @Author : Jclian91
# @File : params.py
# @Place : Minghang, Shanghai
import os
#alt+tab 可以切屏
# 项目文件设置
ProjectDir = os.getcwd()#用于获取当前工作目录（Current Working Directory）的路径。当你在一个脚本中调用 os.getcwd()，它会返回你执行脚本时所在的目录的路径。
Train_File_Path = os.path.join(ProjectDir, 'train.csv')
Test_File_Path = os.path.join(ProjectDir, 'test.csv')
Validation_File_Path=os.path.join(ProjectDir, 'validation.csv')
# 预处理设置
NUM_WORDS = 5500    #这里是词向量大小的数目
PAD = '<PAD>'
PAD_NO = 0
UNK = '<UNK>'
UNK_NO = 1
START_NO = UNK_NO + 1
SENT_LENGTH = 200  #文本长度

# 模型参数
EMBEDDING_SIZE = 300   #詞向量維度300
VALIDATION_BATCH_SIZE=16
TRAIN_BATCH_SIZE = 32   #训练批数大小
TEST_BATCH_SIZE = 16    #测试批数大小
LEARNING_RATE = 0.01  #0.001  #学习率
EPOCHS = 30   #训练周期