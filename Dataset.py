import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class MyDataset(Dataset):#train:布尔值，指定是加载训练集 (True) 还是测试集 (False);test_size：测试集的比例，默认为 0.1667（即 1/6）;random_state：随机种子，用于保证划分结果的可复现性。
    def __init__(self, root_dir, transform=None, train=True, test_size=0.1667, random_state=42):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir  # 根目录
        self.transform = transform
        self.train = train  # 是否是训练集
        self.test_size = test_size  # 测试集比例（1/6 ≈ 0.1667）
        self.random_state = random_state  # 随机种子，保证划分结果可复现

        # 获取所有类别文件夹
        self.classes = sorted(os.listdir(root_dir))  # 按字母顺序排序
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  # 类别到索引的映射

        # 获取所有图片路径和对应的标签
        all_img_paths = []
        all_labels = []
        for idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)  # 类别文件夹路径
            if os.path.isdir(cls_dir):  # 确保是文件夹
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)  # 图片路径
                    all_img_paths.append(img_path)
                    all_labels.append(idx)  # 标签是类别索引

        # 划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_img_paths, all_labels, test_size=self.test_size, random_state=self.random_state, stratify=all_labels
        )

        # 如果是训练集，使用训练数据；如果是测试集，使用测试数据
        if  self.train:
            self.img_list= X_train
            self.labels= y_train
        else:
            self.img_list= X_test
            self.labels= y_test


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = self.labels[item]

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform is not None:
            image = self.transform(image)

        # 将标签转换为张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label
#
#定义数据增强和预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),         # 转换为张量
    transforms.Normalize(          # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 创建数据集实例,创建训练集
dataset = MyDataset(root_dir='D:\\MachineLearning\\GruaduationProject\\PCB_DATASET\\PCB_DATASET\\images', transform=transform,train=True)

# 创建测试集
'''test_dataset = MyDataset(root_dir=r'D:\MachineLearning\GruaduationProject\PCB_DATASET\PCB_DATASET\images',
                         transform=transform, train=False)'''
# 创建数据加载器
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in dataloader:
    print(images.shape)  # 输出：torch.Size([batch_size, 3, 224, 224])第一维表示批量大小（4）。第二维表示通道数（3）。最后两维表示图像的高度和宽度（224x224）。
    print(labels.shape)  # 输出：torch.Size([batch_size])