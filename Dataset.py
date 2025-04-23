import os
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import param
from param import Num_Anchors
from preprocess import rotate_image_and_xml
import numpy as np
import cv2 # 导入 OpenCV 库
from ad import apply_augmentations # 导入你的增强函数
class MyDataset(Dataset):#transform=True表示进行变换，会把他变成张量
    def __init__(self, root_dir, transform=None, train=True, test_size=0.1667, random_state=42, num_anchors=Num_Anchors, num_classes=7,rotate_angle=None,background_class=0):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train  # 标记是否是训练集，通常只在训练时进行增强
        self.test_size = test_size
        self.random_state = random_state
        self.num_anchors = Num_Anchors
        self.num_classes = num_classes
        self.background_class = background_class # 存储背景类别索引# 类别数，假设共有 7 个类别

        # 假设 XML 文件存储在 Annotations 文件夹中，图片存储在 images 文件夹中
        "使用原来的数据集"
        self.annotations_dir = os.path.join(root_dir, 'Annotations')
        self.images_dir = os.path.join(root_dir, 'images')
        "使用旋转后的数据集"
        # self.annotations_dir = os.path.join(root_dir, 'rotated_annotations')
        # self.images_dir = os.path.join(root_dir, 'rotated_images')
        # self.rotate_angle = rotate_angle

        # 获取所有 XML 文件路径（递归遍历子文件夹）
        self.all_xml_paths = []
        for root, dirs, files in os.walk(self.annotations_dir):
            for file in files:
                if file.endswith('.xml'):
                    self.all_xml_paths.append(os.path.join(root, file))

        # 获取对应的图像路径
        self.all_img_paths = []
        for xml_path in self.all_xml_paths:
            # 提取文件名，不包括路径和扩展名
            file_name = os.path.splitext(os.path.basename(xml_path))[0]
            # 构建对应图像路径
            img_sub_dir = os.path.relpath(os.path.dirname(xml_path), self.annotations_dir)
            img_path = os.path.join(self.images_dir, img_sub_dir, f"{file_name}.jpg")
            self.all_img_paths.append(img_path)

        # 过滤掉不存在的图像路径
        valid_indices = [i for i, path in enumerate(self.all_img_paths) if os.path.exists(path)]
        self.all_xml_paths = [self.all_xml_paths[i] for i in valid_indices]
        self.all_img_paths = [self.all_img_paths[i] for i in valid_indices]

        # 划分为训练集和测试集
        if self.train:
            self.img_list, _, self.xml_list, _ = train_test_split(
                self.all_img_paths, self.all_xml_paths, test_size=self.test_size, random_state=self.random_state
            )
        else:
            _, self.img_list, _, self.xml_list = train_test_split(
                self.all_img_paths, self.all_xml_paths, test_size=self.test_size, random_state=self.random_state
            )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        xml_path = self.xml_list[item]

        # 打开图像
        image = Image.open(img_path).convert('RGB')
        original_img_width, original_img_height = image.size

        # 解析 XML 获取对象信息和原始边界框
        objects = self.parse_xml(xml_path)
        # 提取边界框和类别名称到列表中，方便传递给增强函数
        bboxes = [obj['bbox'] for obj in objects]
        class_names = [obj['name'] for obj in objects]


        # 保存原始图像的numpy格式，用于可视化
        original_image_np = np.array(image)
        # 解析 XML 获取对象信息和原始边界框
        objects = self.parse_xml(xml_path)
        # --- 图片处理：保持长宽比缩放 + 填充 ---
        target_short = 1024 # 目标短边尺寸   1024
        padded_size = (1024, 1024) # 最终填充后的尺寸 1024

        # 计算缩放比例（使短边达到 target_short）
        scale = target_short / min(original_img_width, original_img_height)
        new_w, new_h = int(original_img_width * scale), int(original_img_height * scale)

        # 缩放图片，保持长宽比
        # 此时图片尺寸为 (new_w, new_h)，其中一条边是 1024，另一条 >= 1024
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 计算填充量，将缩放后的图片居中放置在 padded_size 画布上
        pad_x = (padded_size[0] - new_w) // 2
        pad_y = (padded_size[1] - new_h) // 2

        # 创建一个新的填充画布 (1024x1024)
        padded_image = Image.new('RGB', padded_size, (114, 114, 114))
        # 将缩放后的图片粘贴到填充画布上（完成填充和可能的长边裁剪）
        padded_image.paste(image, (pad_x, pad_y))

        # 应用变换 (ToTensor, Normalize) 到填充后的图片
        if self.transform is not None:
            transformed_image = self.transform(padded_image)
        else:
             # 如果没有 transform，手动转换为 Tensor (例如用于测试)
             transformed_image = transforms.ToTensor()(padded_image)
             # 如果标准流程包含 Normalize，此处也需要添加
             # transformed_image = transforms.Normalize(mean=[...], std=[...])(transformed_image)

        # --- 标签处理：将 XML 对象转换为模型输入格式 ---
        objects = self.parse_xml(xml_path)
        # 传入图片处理过程中计算的缩放比例和填充偏移量
        y_batch = self.convert_to_model_input(
            objects,
            original_size=(original_img_width, original_img_height),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            padded_size=padded_size # 传入填充后的尺寸 (1024, 1024) 用于归一化
        )

        return transformed_image, y_batch, original_image_np, (original_img_width, original_img_height), scale, pad_x, pad_y, padded_size

    def parse_xml(self, xml_path):
        # 解析 XML 文件，提取目标信息
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text  # 类别名称
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]  #  # 原始像素坐标
            })
        return objects

    def convert_to_model_input(self, objects, original_size, scale, pad_x, pad_y, padded_size):
        # 将目标信息转换为模型的输入格式（`y_batch`）
        # 假设模型使用锚框（anchors）预测目标，每个锚框包含类别和边界框
        # 这里我们生成一个标签张量，形状为 (1, num_anchors, num_classes + 4)
        # 其中，num_anchors 是锚框的数量，num_classes 是类别数

        name_to_index = {
            'background':0,
            'missing_hole': 1,
            'mouse_bite': 2,
            'open_circuit': 3,
            'short': 4,
            'spur': 5,
            'spurious_copper': 6
            # 添加其他类别
        }
        # 可选：断言检查前景类别索引是否有效 (例如 > 背景索引)
        for name, idx in name_to_index.items():
            if name != 'background':
                assert idx > 0, f"前景类别 '{name}' 的索引 {idx} 小于或等于背景索引 {0}"
            assert idx < self.num_classes, f"类别 '{name}' 的索引 {idx} 大于或等于总类别数 {self.num_classes}"

        # 初始化标签张量
        y_batch = torch.zeros(( self.num_anchors, self.num_classes + 4))  # batch_size=1
        y_batch[:, 0] = 1.0  # 默认所有锚框为背景
        padded_width, padded_height = padded_size  # 应该是 (1024, 1024)
        # 原始图像尺寸和缩放后图像尺寸
        for i, obj in enumerate(objects):
            if i >= self.num_anchors:
                # 如果对象数量超过 Anchor 数量，打印警告并停止填充
                # print(f"警告: 图片 '{os.path.basename(self.img_list[item])}' 中的对象数量 ({len(objects)}) 超过 Anchor 数量 ({self.num_anchors})。索引大于等于 {self.num_anchors} 的对象将被忽略。")
                break  # 超过 Anchor 数量，停止填充

            name = obj['name']
            original_bbox = obj['bbox'] # 原始像素坐标 [xmin, ymin, xmax, ymax]

            # 获取类别索引，如果名称未找到，默认为背景
            class_idx = name_to_index.get(name, self.background_class)

            # 如果类别是背景，或者类别索引超出预期范围，则跳过此对象
            # 通常 XML 中的对象不应是背景
            if class_idx == self.background_class:
                 # print(f"警告: 跳过背景对象 '{name}'。")
                 continue # 跳过背景对象

                 # --- 边界框坐标转换 ---
                 # 1. 使用与图片缩放相同的比例对原始 bbox 进行缩放
            xmin_scaled = original_bbox[0] * scale
            ymin_scaled = original_bbox[1] * scale
            xmax_scaled = original_bbox[2] * scale
            ymax_scaled = original_bbox[3] * scale

            # 2. 考虑填充的偏移量，添加填充量
            xmin_padded = xmin_scaled + pad_x
            ymin_padded = ymin_scaled + pad_y
            xmax_padded = xmax_scaled + pad_x
            ymax_padded = ymax_scaled + pad_y

            # 3. 归一化到填充后的尺寸 (1024, 1024) [0, 1] 范围
            # 在归一化之前，将坐标限制在填充后的边界内 [0, padded_size]
            # 这可以处理原始框可能略微超出图片边界或浮点精度问题导致的越界
            xmin_padded = np.clip(xmin_padded, 0, padded_width)
            ymin_padded = np.clip(ymin_padded, 0, padded_height)
            xmax_padded = np.clip(xmax_padded, 0, padded_width)
            ymax_padded = np.clip(ymax_padded, 0, padded_height)

            normalized_bbox = [
                xmin_padded / padded_width,
                ymin_padded / padded_height,
                xmax_padded / padded_width,
                ymax_padded / padded_height
            ]

            # 可选：检查归一化后的边界框是否有效 (宽度和高度 > 0)
            # 如果原始框有效且没有被填充/裁剪完全“压扁”，则归一化后应仍满足 xmin < xmax 且 ymin < ymax
            if normalized_bbox[0] >= normalized_bbox[2] or normalized_bbox[1] >= normalized_bbox[3]:
                # print(f"警告: 对象 '{name}' 在图片 '{os.path.basename(self.img_list[item])}' 中转换后的边界框无效: {normalized_bbox}。原始框: {original_bbox}。缩放比例: {scale}, 填充: ({pad_x}, {pad_y})。跳过此对象。")
                # 如果框变得无效，则不将其赋值给 y_batch。
                # 对应的 Anchor 位置将保持为背景 (默认值)。
                continue  # 跳过此对象如果其框在转换后无效

            # --- 赋值给 y_batch ---
            # 清除此 Anchor 默认的背景标签
            y_batch[i, 0] = 0.0
            # 设置实际的类别标签 (One-Hot 编码)
            y_batch[i, class_idx] = 1.0  # 设置实际类别

            # 设置归一化后的边界框坐标
            y_batch[i, self.num_classes:] = torch.tensor(normalized_bbox, dtype=torch.float32)

        # 注意: 图片中对象数量少于 Anchor 数量时，多余的 Anchor 将保持为背景 (默认初始化值)。
        # 因无效框而被跳过的对象，其对应的 Anchor 也将保持为背景。
        # 这对于训练分类头区分负样本是正确的。

        return y_batch


#测试用例
def test_Dataset():
    # 定义数据增强和预处理操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(  # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 创建数据集实例,返回的image是通过transform的一个张量
    dataset = MyDataset(root_dir=param.root_dir, transform=transform, train=True)

    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 获取一个批次的数据并输出 y_batch 的细节

    # 下面适用于转换为张量的情况
    for images, y_batch in data_loader:
        print("Image shape:", images.shape)  # 输出图像的形状，Image shape: torch.Size([4, 3, 224, 224])
        print("y_batch shape:", y_batch.shape)  # 输出 y_batch 的形状
        print("y_batch content:", y_batch)  # 输出 y_batch 的内容
        break  # 只获取一个批次的数据


if __name__ == "__main__":
    test_Dataset()