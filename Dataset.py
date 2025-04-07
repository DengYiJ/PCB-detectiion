import os
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import param
from preprocess import rotate_image_and_xml
import numpy as np
class MyDataset(Dataset):#transform=True表示进行变换，会把他变成张量
    def __init__(self, root_dir, transform=None, train=True, test_size=0.1667, random_state=42, num_anchors=6, num_classes=6,rotate_angle=None):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.num_anchors = num_anchors
        self.num_classes = num_classes  # 类别数，假设共有 6 个类别

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

        # 保存原始图像的numpy格式，用于可视化
        original_image_np = np.array(image)
        # 解析 XML 文件，获取目标信息
        objects = self.parse_xml(xml_path)

        # 转换为目标检测模型的输入格式
        y_batch = self.convert_to_model_input(objects, original_img_width, original_img_height)

        # 应用变换
        if self.transform is not None:
            transformed_image = self.transform(image)
        #y_batch = torch.zeros(1, 9, 10)
        #y_batch[0, 0, :] = [1.0, 0, 0, 0, 0, 0, 2459, 1274, 2530, 1329]
        # y_batch[0, 1, :] = [1.0, 0, 0, 0, 0, 0, 1613, 334, 1679, 396]
        # y_batch[0, 2, :] = [1.0, 0, 0, 0, 0, 0, 1726, 794, 1797, 854]
        #第一0是batch中第几个，这里初始化的是每轮1个，也就是第一个；如果是每轮4个处理，那就是4个图片中的第一张。第二个是9个锚框，数量自己设计；第三个10是6（类别概率）+4（边界框）
        return transformed_image, y_batch,original_image_np, (original_img_width, original_img_height)

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
                'bbox': [xmin, ymin, xmax, ymax]
            })
        return objects

    def convert_to_model_input(self, objects,img_width,img_height):
        # 将目标信息转换为模型的输入格式（`y_batch`）
        # 假设模型使用锚框（anchors）预测目标，每个锚框包含类别和边界框
        # 这里我们生成一个标签张量，形状为 (1, num_anchors, num_classes + 4)
        # 其中，num_anchors 是锚框的数量，num_classes 是类别数

        name_to_index = {
            'missing_hole': 0,
            'mouse_bite': 1,
            'open_circuit': 2,
            'short': 3,
            'spur': 4,
            'spurious_copper': 5
            # 添加其他类别
        }

        # 初始化标签张量
        y_batch = torch.zeros(( self.num_anchors, self.num_classes + 4))  # batch_size=1

        # 原始图像尺寸和缩放后图像尺寸
        original_width = img_width
        original_height = img_height
        resized_width = 1024
        resized_height = 1024

        # 缩放比例
        width_scale = resized_width / original_width
        height_scale = resized_height / original_height

        #print(f"Processing objects: {objects}")  # 调试信息
        # 填充标签
        for i, obj in enumerate(objects):
            if i >= self.num_anchors:
                break  # 超过锚框数量，停止填充

            name = obj['name']
            bbox = obj['bbox']

            # 设置分类标签（one-hot 编码）
            class_idx = name_to_index.get(name, self.num_classes)  # 默认是最末尾的索引
            #print(f"Object {i}: name={name}, class_idx={class_idx}")  # 调试信息

            if class_idx < self.num_classes:
                y_batch[ i, class_idx] = 1.0  # 类别概率

                # 设置边界框坐标并归一化
                xmin, ymin, xmax, ymax = bbox
                # 调整边界框坐标到缩放后的图像尺寸
                xmin = xmin * width_scale
                ymin = ymin * height_scale
                xmax = xmax * width_scale
                ymax = ymax * height_scale

                # 归一化到 [0, 1] 范围
                normalized_bbox = [
                    xmin / resized_width,  # xmin
                    ymin / resized_height,  # ymin
                    xmax / resized_width,  # xmax
                    ymax / resized_height  # ymax
                ]
            #    print(f"imgW={img_width}, imgH={img_height}")
            #     print(f"Object {i}: Original bbox={bbox}, Normalized bbox={normalized_bbox}")
                for coord in normalized_bbox:
                    assert isinstance(coord, float), "normalized_bbox 中的元素不是浮点数"

                y_batch[i, self.num_classes:] = torch.tensor(normalized_bbox)

            # 设置边界框坐标
          #  y_batch[i, self.num_classes:] = torch.tensor(bbox)  # [xmin, ymin, xmax, ymax]

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