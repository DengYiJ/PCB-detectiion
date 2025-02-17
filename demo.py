from torch.utils.data import DataLoader, Dataset

import torch
class MyDataset(Dataset):
    def __init__(self):
        # 假设我们有一些数据
        self.data = [1, 2, 3, 4, 5]
        self.targets = [10, 20, 30, 40, 50]#一组预定义的标签列表

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 返回一个样本及其标签
        return self.data[idx], self.targets[idx]

# 创建数据集实例
dataset = MyDataset()

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 迭代数据加载器
for x_batch, y_batch in data_loader:
    print(f"Features: {x_batch}, Labels: {y_batch}")

    import torch

    print(torch.__version__)