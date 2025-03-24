import os
#alt+tab 可以切屏
# 项目文件设置
ProjectDir = os.getcwd()#用于获取当前工作目录（Current Working Directory）的路径。当你在一个脚本中调用 os.getcwd()，它会返回你执行脚本时所在的目录的路径。
Train_File_Path = os.path.join(ProjectDir, 'train.csv')
Test_File_Path = os.path.join(ProjectDir, 'test.csv')
Validation_File_Path=os.path.join(ProjectDir, 'validation.csv')
# root_dir='D:\\MachineLearning\\GruaduationProject\\PCB_DATASET\\PCB_DATASET'
root_dir='./PCB_DATASET'
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
VALIDATION_BATCH_SIZE=4
TRAIN_BATCH_SIZE = 4   #训练批数大小
TEST_BATCH_SIZE = 4    #测试批数大小
LEARNING_RATE = 0.00001  #0.001  #学习率
EPOCHS = 30   #训练周期
Droprate=0.5
Embeding_dim=32
Netdepth=2