import os
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import param
from param import Num_Anchors ,visualizations_dir
from preprocess import rotate_image_and_xml
import numpy as np
import cv2 # 导入 OpenCV 库
from ad import apply_augmentations # 导入你的增强函数
from VisualizedPredict import decode_boxes
name_to_index = {
    'background': 0,
    'missing_hole': 1,
    'mouse_bite': 2,
    'open_circuit': 3,
    'short': 4,
    'spur': 5,
    'spurious_copper': 6
    # 添加其他类别
}
def compute_iou_matrix(boxes1, boxes2):
    """
    计算两组边界框之间的 IoU 矩阵。
    boxes1: shape (N, 4) 或 (N, 4), 格式 [xmin, ymin, xmax, ymax]
    boxes2: shape (M, 4) 或 (M, 4), 格式 [xmin, ymin, xmax, ymax]
    返回: shape (N, M) 的 IoU 矩阵
    """
    # 将输入转换为 torch 张量 (如果不是)
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)

    # 计算交集区域
    # 找到每个框的左上角和右下角坐标的最大值和最小值
    # (N, 1, 4) vs (1, M, 4)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # 左上角 (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # 右下角 (N, M, 2)

    # 计算交集框的宽度和高度
    wh = (rb - lt).clamp(min=0) # 形状 (N, M, 2)，小于零的设为 0
    inter = wh[:, :, 0] * wh[:, :, 1] # 交集面积 (N, M)

    # 计算两个框的面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0) # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0) # (M,)

    # 计算并集区域
    area1 = area1[:, None] # (N, 1)
    area2 = area2[None, :] # (1, M)
    union = area1 + area2 - inter # (N, M)

    # 计算 IoU
    iou = inter / (union + 1e-8) # 加一个小的 epsilon 防止除以零

    return iou

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

        # # --- 在填充/划分 self.img_list 和 self.xml_list 之后添加以下打印 ---
        # print("\n--- Debug Print: File lists after initialization and split ---")
        # print(f"self.img_list[:10]: {self.img_list[:min(len(self.img_list), 10)]}")  # 打印前10个图片路径
        # print(f"self.xml_list[:10]: {self.xml_list[:min(len(self.xml_list), 10)]}")  # 打印前10个XML路径
        # print("----------------------------------------------------------------")
        # # --- 结束调试打印 ---

        # --- 定义你的 num_anchors 个参考框 (Reference Boxes) ---
        # 使用 K-Means 结果和平均缩放比例来计算这些框的归一化坐标。

        # 你从 K-Means 得到的代表性 Anchor 尺寸 (原始像素)
        representative_anchor_sizes_wh_orig_pixels = np.array([
            [46.10, 44.34], [63.72, 66.93], [94.89, 54.05],
            [51.42, 102.90], [147.64, 57.47], [101.89, 98.06]
        ], dtype=np.float32)
        # representative_anchor_sizes_wh_orig_pixels = np.array([
        #     [43.23, 41.82], [60.93, 61.48], [49.27, 84.75],
        #     [80.65, 51.90], [111.35, 56.53], [54.10,120.50],[84.42,85.20],[162.62,53.38],[124.97,113.58]
        # ], dtype=np.float32)

        num_shapes_per_location = representative_anchor_sizes_wh_orig_pixels.shape[0]  # 应该是 6
        # 你计算出的训练集平均预处理缩放比例
        # 请将你的平均缩放比例填在这里
        average_preprocessing_scale = 0.3724  # <--- 你的平均缩放比例结果

        # 目标图片尺寸 (填充后)
        target_image_dim = 1024
        grid_size = 64  # 16x16 的特征图网格 同步修改param.Num_Anchors
        # 计算 Anchor 网格步长在 1024x1024 像素空间的大小
        grid_stride = target_image_dim // grid_size  # 1024 / 32 = 32

        # 检查 Num_Anchors 是否与预期一致
        expected_num_anchors = grid_size * grid_size * num_shapes_per_location
        if self.num_anchors != expected_num_anchors:
            print(
                f"Warning: Num_Anchors ({self.num_anchors}) does not match expected number from grid and shapes ({expected_num_anchors}).")
            # 你可能需要根据你的 num_shapes_per_location 调整 Num_Anchors 在 param.py 中的值

        # 计算每个代表性 Anchor 形状在 1024x1024 填充图上的归一化尺寸 (0-1)
        normalized_anchor_sizes_wh_padded = (representative_anchor_sizes_wh_orig_pixels * average_preprocessing_scale) / target_image_dim  # 形状 (num_shapes_per_location, 2)
        # 计算 16x16 网格点的归一化中心坐标 (0-1)
        # 网格步长在 1024x1024 像素空间是 1024 / 16 = 64
        # 网格点中心是 (j * 64 + 32, i * 64 + 32)
        # 归一化到 0-1: ((j * 64 + 32) / 1024, (i * 64 + 32) / 1024)
        grid_centers_normalized = []
        for i in range(grid_size): # 行 (Y)
            for j in range(grid_size): # 列 (X)
                cx_norm = (j * grid_stride + grid_stride / 2) / target_image_dim# <--- 基于1024x1024归一化的中心点坐标
                cy_norm = (i  * grid_stride + grid_stride / 2) / target_image_dim
                grid_centers_normalized.append([cx_norm, cy_norm])
        grid_centers_normalized = torch.tensor(grid_centers_normalized, dtype=torch.float32) # 形状 (grid_size*grid_size, 2)

        # 生成所有的参考框 (self.reference_boxes)
        # 总数量 Num_Anchors = grid_size * grid_size * num_shapes_per_location
        self.reference_boxes = torch.zeros(self.num_anchors, 4, dtype=torch.float32) # 形状 (Num_Anchors, 4)

        anchor_idx = 0
        for center_norm in grid_centers_normalized: # 遍历所有网格中心 (256个)
            cx, cy = center_norm
            for shape_idx in range(num_shapes_per_location): # 遍历所有 Anchor 形状 (6个)
                norm_w, norm_h = normalized_anchor_sizes_wh_padded[shape_idx]

                # 计算 Anchor Box 的归一化 [xmin, ymin, xmax, ymax] 坐标
                xmin = cx - norm_w / 2
                ymin = cy - norm_h / 2
                xmax = cx + norm_w / 2
                ymax = cy + norm_h / 2

                # # 裁剪坐标到 [0, 1] 范围 (防止 Anchor 超出图片边界)
                # self.reference_boxes[anchor_idx, 0] = torch.clamp(torch.tensor(xmin), 0.0, 1.0)
                # self.reference_boxes[anchor_idx, 1] = torch.clamp(torch.tensor(ymin), 0.0, 1.0)
                # self.reference_boxes[anchor_idx, 2] = torch.clamp(torch.tensor(xmax), 0.0, 1.0)
                # self.reference_boxes[anchor_idx, 3] = torch.clamp(torch.tensor(ymax), 0.0, 1.0)
                # 将计算出的坐标打包成一个张量，并整体进行裁剪
                anchor_coords = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                self.reference_boxes[anchor_idx, :] = torch.clamp(anchor_coords, 0.0, 1.0)
                anchor_idx += 1

        # 确认生成的参考框数量正确
        assert anchor_idx == self.num_anchors, "Mismatch in generated reference boxes count!"
        # print(f"Generated {self.num_anchors} reference boxes.")
        # --- 添加基本验证 ---
        # print("\n--- Basic Verification of self.reference_boxes ---")
        # print(f"Shape of self.reference_boxes: {self.reference_boxes.shape}")
        # 检查是否是 torch 张量
        assert isinstance(self.reference_boxes, torch.Tensor), "self.reference_boxes should be a torch.Tensor"
        # 检查数据类型是否正确
        assert self.reference_boxes.dtype == torch.float32, "self.reference_boxes dtype should be torch.float32"
        # 打印前几个参考框的坐标，检查是否在 [0, 1] 范围内
        # print("First 5 reference boxes (normalized [xmin, ymin, xmax, ymax]):")
        # print(self.reference_boxes[:5])
        # 打印后几个参考框的坐标
        # print("Last 5 reference boxes (normalized [xmin, ymin, xmax, ymax]):")
        # print(self.reference_boxes[-5:])
        # 检查所有坐标是否都在 [0, 1] 范围内 (由于裁剪，应该如此)
        assert torch.all(self.reference_boxes >= 0.0) and torch.all(
            self.reference_boxes <= 1.0), "Reference box coordinates out of [0, 1] range!"
        # print("Basic verification passed: Reference box shape and sample values look reasonable.")
        # print("----------------------------------------------------")

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
        target_size = 1024 # 目标尺寸   1024
        padded_size = (1024, 1024) # 最终填充后的尺寸 1024

        # 计算缩放比例（使短边达到 target_short）
        # scale = target_short / min(original_img_width, original_img_height)
        scale = target_size / max(original_img_width, original_img_height)
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
        y_batch_xywh,gt_boxes_padded_norm = self.convert_to_model_input(
            objects,
            original_size=(original_img_width, original_img_height),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            padded_size=padded_size # 传入填充后的尺寸 (1024, 1024) 用于归一化
        )

        return transformed_image, y_batch_xywh, objects,original_image_np, (original_img_width, original_img_height), scale, pad_x, pad_y, padded_size,gt_boxes_padded_norm

    def parse_xml(self, xml_path):
        """
        解析 XML 文件，提取目标信息 (包括类别名称、索引和边界框)。

        Args:
            xml_path (str): XML 文件的完整路径。

        Returns:
            list: 包含文件中每个目标信息的字典列表，每个字典包含 'name', 'bbox', 'class_idx'。
        """
        # --- Debug Print: Show which XML file is being processed ---
        # print(f"    parse_xml: Processing file received: {xml_path}")
        # --- End Debug Print ---
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text  # 类别名称

            # --- Debug Print: Check name read from XML ---
            # print(f"    parse_xml: Read name from XML: '{name}' (type: {type(name)}) for file: {os.path.basename(xml_path)})")  # 打印读取到的名字及其类型
            # --- End Debug Print ---

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # 根据类别名称查找类别索引

            # 确保类别名称在 name_to_index 字典中存在
            class_idx = name_to_index.get(name)  # 使用 .get() 方法，如果名称不存在，返回 None
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax] ,  # 原始像素坐标
                'class_idx': class_idx  # 添加类别索引
            })
        return objects

    def convert_to_model_input(self, objects, original_size, scale, pad_x, pad_y, padded_size):
        """
              将真实目标框信息转换为模型训练所需的标签 y_batch。
              实现真实框到参考框 (Anchor) 的匹配。

              Args:
                  objects (list): 包含图片中真实目标信息的列表 (解析 XML 得到)。
                                  每个元素是一个字典，包含 'bbox' (原始像素坐标), 'class_name', 'class_idx'。
                  original_size (tuple): 原始图片尺寸 (width, height)。
                  padded_size (tuple): 填充后的图片尺寸 (width, height)。
                  scale (float): 图像缩放比例。
                  pad_x (int): X 方向的填充量。
                  pad_y (int): Y 方向的填充量。

              Returns:
                  torch.Tensor: 形状为 (num_anchors, num_classes + 4) 的标签张量 y_batch。
                                对于匹配到的正样本 Anchor，包含类别 One-Hot 编码和真实框归一化坐标。
                                对于负样本 Anchor，包含背景类别 One-Hot 编码。
        """
        num_objects = len(objects)
        num_anchors = self.num_anchors
        num_classes = self.num_classes
        padded_width, padded_height = padded_size

        # 步骤 1: 初始化 y_batch (所有 Anchor 默认标记为背景)
        # y_batch 的形状是 (Num_Anchors, num_classes + 4)
        y_batch = torch.zeros(num_anchors, num_classes + 4, dtype=torch.float32,
                              device=self.reference_boxes.device)  # 确保在同一设备
        # 设置背景类别为 1.0 (假设背景类索引为 0)
        y_batch[:, self.background_class] = 1.0
        # 如果图片没有真实目标，所有 Anchor 都是背景，直接返回
        if num_objects == 0:
            print("    No objects found in XML. y_batch remains all background.")
            return y_batch

        # 步骤 2: 准备真实目标框和类别 (转换为归一化到填充后尺寸的坐标)
        # 存储所有有效真实目标框的归一化坐标和类别 One-Hot
        gt_boxes_padded_norm = []  # 存储归一化到 padded_size 的 [xmin, ymin, xmax, ymax]
        gt_classes_one_hot = []  # 存储真实类别的 One-Hot 编码
        valid_original_objects = []  # <-- 在这里创建列表，存储转换后有效的原始对象信息 (用于调试)

        padded_width, padded_height = padded_size  # 应该是 (1024, 1024)
        # 原始图像尺寸和缩放后图像尺寸
        for i, obj in enumerate(objects):
            if i >= self.num_anchors:
                # 如果对象数量超过 Anchor 数量，打印警告并停止填充
                # print(f"警告: 图片 '{os.path.basename(self.img_list[item])}' 中的对象数量 ({len(objects)}) 超过 Anchor 数量 ({self.num_anchors})。索引大于等于 {self.num_anchors} 的对象将被忽略。")
                break  # 超过 Anchor 数量，停止填充

            name = obj['name']
            original_bbox = obj['bbox'] # 原始像素坐标 [xmin, ymin, xmax, ymax]
            # # --- 添加调试打印 ---
            # print(f"    Processing original bbox: {original_bbox} for class {name}")
            # print(f"    Preprocessing params: scale={scale:.4f}, pad_x={pad_x}, pad_y={pad_y}, padded_size={padded_size}")
            # # --- 结束调试打印 ---
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
            # # --- 添加调试打印 ---
            # print(f"    Padded pixel coords (before normalization): [{xmin_padded:.2f}, {ymin_padded:.2f}, {xmax_padded:.2f}, {ymax_padded:.2f}]")
            # # --- 结束调试打印 ---
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
            # # --- 添加调试打印 ---
            # print(f"    Normalized bbox (0-1 in padded_size): {normalized_bbox}")
            # # --- 结束调试打印 ---
            # 可选：检查归一化后的边界框是否有效 (宽度和高度 > 0)
            # 如果原始框有效且没有被填充/裁剪完全“压扁”，则归一化后应仍满足 xmin < xmax 且 ymin < ymax
            if normalized_bbox[0] >= normalized_bbox[2] or normalized_bbox[1] >= normalized_bbox[3]:
                # print(f"警告: 对象 '{name}' 在图片 '{os.path.basename(self.img_list[item])}' 中转换后的边界框无效: {normalized_bbox}。原始框: {original_bbox}。缩放比例: {scale}, 填充: ({pad_x}, {pad_y})。跳过此对象。")
                print(f"Warning: Invalid ground truth box after preprocessing: {original_bbox} -> {normalized_bbox}. Skipping.")
                continue  # 跳过此对象如果其框在转换后无效

            gt_boxes_padded_norm.append(normalized_bbox)
            # 转换为 One-Hot 编码
            one_hot_class = torch.zeros(num_classes, dtype=torch.float32)
            one_hot_class[class_idx] = 1.0
            gt_classes_one_hot.append(one_hot_class)
            valid_original_objects.append({'bbox_norm': normalized_bbox, 'class_name':name, 'class_idx': class_idx})  # <-- 存储有效对象的原始信息

        # 如果没有有效真实目标框，所有 Anchor 都是背景，直接返回，如果有有效真实目标框，转换为 PyTorch张量并继续处理
        if len(gt_boxes_padded_norm) == 0:
             print("    No valid objects found after preprocessing. y_batch remains all background.")
             return y_batch
        # 将真实目标框和类别信息转换为 PyTorch 张量
        gt_boxes_padded_norm = torch.tensor(gt_boxes_padded_norm, dtype=torch.float32,
                                            device=y_batch.device)  # 形状 (num_valid_objects, 4)
        gt_classes_one_hot = torch.stack(gt_classes_one_hot).to(y_batch.device)  # 形状 (num_valid_objects, num_classes)
        num_valid_objects = gt_boxes_padded_norm.shape[0]  # 实际有效的真实目标数量

        # # --- Debug Print: Check valid GT class information before matching ---
        # print("    Valid GT Class Information (before matching):")
        # for i, obj_info in enumerate(valid_original_objects):
        #     print(f"      Valid Index {i}: Original Name: {obj_info['class_name']}, Class Index: {obj_info['class_idx']}, Normalized Bbox: {obj_info['bbox_norm']}")
        # # --- End Debug Print ---

        # 步骤 3: 计算 IoU (有效真实框 vs 参考框)
        # self.reference_boxes 形状 (num_anchors, 4)，已在 __init__ 中定义并归一化到 0-1 (padded_size)
        # gt_boxes_padded_norm 形状 (num_valid_objects, 4)，已归一化到 0-1 (padded_size)
        # iou_matrix 形状 (num_valid_objects, num_anchors)
        iou_matrix = compute_iou_matrix(gt_boxes_padded_norm, self.reference_boxes)

        # --- 添加调试打印：检查 IoU 矩阵 ---
        # print(f"    IoU matrix shape: {iou_matrix.shape}")

        positive_iou_threshold=0.5

        # 步骤 4: 执行匹配算法 (将真实框匹配到参考框/Anchor)
        # 这是核心逻辑，需要根据你的匹配策略填充 matched_gt_indices 和 best_ious_for_anchor
        # matched_gt_indices: 形状 (num_anchors,), 存储每个 Anchor 匹配到的真实目标索引 (在 gt_boxes_padded_norm 中的索引, -1 表示未匹配)
        # best_ious_for_anchor: 形状 (num_anchors,), 存储每个 Anchor 与匹配到的 GT 的最高 IoU

        matched_gt_indices = torch.full((num_anchors,), -1, dtype=torch.long, device=y_batch.device) # 初始化匹配结果：所有 Anchor 默认不匹配任何 GT (-1)
        best_ious_for_anchor = torch.zeros(num_anchors, dtype=torch.float32, device=y_batch.device) # 初始化最佳 IoU：所有 Anchor 的最佳 IoU 默认是 0

        # 找到每个 Anchor 与哪个真实目标的 IoU 最高 (用于策略 2 和解决策略 1 冲突)
        max_iou_for_anchor, best_gt_indices_for_anchor = iou_matrix.max(dim=0)  # 形状 (num_anchors,)

        if num_valid_objects > 0:
            # 找到每个 GT 匹配的最佳 Anchor (按 GT 维度求最大值)
            max_iou_for_gt, best_anchor_indices_for_gt = iou_matrix.max(dim=1)  # 形状 (num_valid_objects,)

            # --- Debug Print: Check GT's best Anchor IoU ---
            # print("    Max IoU for each Ground Truth and best Anchor index:")
            # for gt_idx in range(num_valid_objects):
            #     # Use valid_original_objects to get original info for printing
            #     print(f"      GT {gt_idx} (Name: '{valid_original_objects[gt_idx]['class_name']}', Index: {valid_original_objects[gt_idx]['class_idx']}): Max IoU = {max_iou_for_gt[gt_idx]:.4f}, Best Anchor Index = {best_anchor_indices_for_gt[gt_idx]}")
            # --- End Debug Print ---

            # 策略 1: 对于每个真实目标，找到与其 IoU 最高的 Anchor，强制将该 Anchor 标记为与此 GT 匹配。
            # 解决一个 Anchor 被多个 GT 认为是最佳匹配的冲突：保留 IoU 最高的匹配。
            for gt_idx in range(num_valid_objects):
                anchor_idx = best_anchor_indices_for_gt[gt_idx]
                iou = max_iou_for_gt[gt_idx]

                # 如果当前 GT 与其最佳 Anchor 的 IoU，比这个 Anchor 目前记录的最佳 IoU 更高，则更新匹配
                # 使用一个小的 epsilon 来处理浮点比较的边界情况
                if iou > best_ious_for_anchor[anchor_idx] + 1e-6:
                    best_ious_for_anchor[anchor_idx] = iou
                    matched_gt_indices[anchor_idx] = gt_idx
                # else: 如果当前 IoU 不更高，说明这个 Anchor 已经被一个 IoU 更高的 GT 强制匹配了，无需处理

            # 策略 2: 对于剩余的 Anchor，如果其与任何真实目标的 IoU 高于正样本阈值，则与 IoU 最高的 GT 匹配。
            # 这个逻辑需要检查 max_iou_for_anchor >= positive_iou_threshold 的 Anchor。
            # 但要避免覆盖策略 1 已经确定的、IoU 更高的强制匹配。

            # 遍历所有 Anchor，基于它们各自的最佳 IoU 来决定匹配
            # 找到所有 IoU >= positive_iou_threshold 的 Anchor 的索引
        high_iou_anchor_mask = max_iou_for_anchor >= positive_iou_threshold
        high_iou_anchor_indices = torch.where(high_iou_anchor_mask)[0]

        # 对于这些高 IoU 的 Anchor，如果它们目前记录的最佳 IoU (best_ious_for_anchor)
        # 小于或等于 (或接近) 它们与自己 IoU 最高的 GT 的 IoU (max_iou_for_anchor)，
        # 则认为它们可以匹配这个 IoU 最高的 GT。
        # 这个判断 `max_iou_for_anchor[anchor_idx] > (best_ious_for_anchor[anchor_idx] - 1e-6)`
        # 实际上是在检查这个 Anchor 的最佳 IoU (来自 max_iou_for_anchor) 是否比策略 1 强制匹配给它的 IoU 更高。
        # 我们只需要确保，如果一个 Anchor 的最佳 IoU 达到阈值，并且它**还没有**被一个更高 IoU 的 GT 强制匹配覆盖，就将其标记为正样本。
        # 使用 best_ious_for_anchor 来跟踪已经被策略 1 或策略 2 确定的“最高 IoU”。

        for anchor_idx in range(num_anchors):
            iou = max_iou_for_anchor[anchor_idx]  # 当前 Anchor 与其最佳 GT 的 IoU
            gt_idx = best_gt_indices_for_anchor[anchor_idx]  # 当前 Anchor 的最佳 GT 索引

            # 如果当前 Anchor 与其最佳 GT 的 IoU >= 正样本阈值
            # 并且 当前 IoU 比这个 Anchor 目前记录的最佳 IoU (best_ious_for_anchor) 更高 (或接近)
            # 这个条件是为了确保不被策略 1 中 IoU 更高的强制匹配覆盖
            if iou >= positive_iou_threshold and iou > (best_ious_for_anchor[anchor_idx] - 1e-6):
                best_ious_for_anchor[anchor_idx] = iou
                matched_gt_indices[anchor_idx] = gt_idx
            # else: IoU 小于阈值，或者已经被一个更高 IoU 的 GT 强制匹配覆盖了，不标记为正样本

        # --- 添加调试打印：检查匹配算法结果 ---
        # print("\n    --- Matching Algorithm Results (After Strategy) ---")
        # 打印匹配到的真实目标索引和最佳 IoU (只打印前 20 个 Anchor 的结果)
        # print("    First 20 Anchors' matching results (matched_gt_indices, best_ious_for_anchor):")
        # # 将两个张量堆叠起来方便查看
        # print(torch.stack([matched_gt_indices[:20].float(), best_ious_for_anchor[:20]], dim=1))

        # 统计最终被标记为匹配到 GT 的 Anchor 数量
        num_matched_anchors_final = torch.sum(matched_gt_indices != -1).item()
        # print(f"    Number of Anchors matched to any GT after final strategy: {num_matched_anchors_final}")

        # 统计最终将被标记为正样本的 Anchor 数量 (IoU >= positive_iou_threshold)
        # 这是从最终的 best_ious_for_anchor 中检查的
        final_positive_anchor_mask = best_ious_for_anchor >= positive_iou_threshold
        num_final_positive_anchors = torch.sum(final_positive_anchor_mask).item()
        # print(f"    Number of Anchors with final best IoU >= {positive_iou_threshold}: {num_final_positive_anchors}")

        # print("    ------------------------------------")
        # --- 结束调试打印 ---

        # 步骤 5: 根据匹配结果和 IoU 阈值填充 y_batch
        # 设定正样本 IoU 阈值 (与损失函数中的匹配阈值一致，通常为 0.5)前面设置过了
        # positive_iou_threshold = 0.5

        # 设定负样本 IoU 阈值 (IoU < 这个阈值被视为负样本，通常为 0.4 或 0.5)
        # IoU 在 negative_iou_threshold 和 positive_iou_threshold 之间的通常被忽略
        negative_iou_threshold = 0.4

        # 遍历所有 Anchor (num_anchors 个预测位置)
        # y_batch 默认是背景，所以我们只需要处理正样本和明确的负样本/忽略样本
        for anchor_idx in range(num_anchors):
            best_iou = best_ious_for_anchor[anchor_idx]
            matched_gt_idx = matched_gt_indices[anchor_idx]

            # 判断当前 Anchor 是否是正样本
            if matched_gt_idx != -1 and best_iou >= positive_iou_threshold:
               #--------------------xyxy版本----------------------
               #  # 是正样本：填充真实类别和边界框
               #  # 清除默认的背景标签
               #  # print(f"      Populating y_batch for positive anchor {anchor_idx}: matched_gt_idx = {matched_gt_idx}, Class Index from valid_original_objects = {valid_original_objects[matched_gt_idx]['class_idx']}")
               #  # print(f"      Populating y_batch for positive anchor {anchor_idx}: One-Hot vector from gt_classes_one_hot = {gt_classes_one_hot[matched_gt_idx]}")
               #  y_batch[anchor_idx, self.background_class] = 0.0
               #  # 设置真实类别 (One-Hot 编码)
               #  y_batch[anchor_idx, :num_classes] = gt_classes_one_hot[matched_gt_idx]
               #  # 设置真实边界框坐标 (归一化到填充后尺寸)
               #  y_batch[anchor_idx, num_classes:] = gt_boxes_padded_norm[matched_gt_idx]
                #--------------------xyxy版本 - ---------------------
                # --------------------xywh版本----------------------
               # 是正样本：填充真实类别和边界框
               # 清除默认的背景标签
               y_batch[anchor_idx, self.background_class] = 0.0
               # 设置真实类别 (One-Hot 编码)
               y_batch[anchor_idx, :num_classes] = gt_classes_one_hot[matched_gt_idx]

               # --- 计算真实框相对于匹配到的 Anchor Box 的偏移量 [gt_dx, gt_dy, gt_dw, gt_dh] ---
               # 获取匹配到的真实框 (归一化到填充后尺寸)
               gt_box_padded_norm = gt_boxes_padded_norm[matched_gt_idx]  # 形状 (4,) [xmin, ymin, xmax, ymax]

               # 获取对应的 Anchor Box (归一化到填充后尺寸)
               anchor_box = self.reference_boxes[anchor_idx]  # 形状 (4,) [xmin, ymin, xmax, ymax]

               # 将真实框和 Anchor Box 转换为中心点和宽高格式 [cx, cy, w, h]
               gt_cx = (gt_box_padded_norm[0] + gt_box_padded_norm[2]) / 2
               gt_cy = (gt_box_padded_norm[1] + gt_box_padded_norm[3]) / 2
               gt_w = gt_box_padded_norm[2] - gt_box_padded_norm[0]
               gt_h = gt_box_padded_norm[3] - gt_box_padded_norm[1]

               anchor_cx = (anchor_box[0] + anchor_box[2]) / 2
               anchor_cy = (anchor_box[1] + anchor_box[3]) / 2
               anchor_w = anchor_box[2] - anchor_box[0]
               anchor_h = anchor_box[3] - anchor_box[1]

               # 计算偏移量 [gt_dx, gt_dy, gt_dw, gt_dh]
               # 避免除以零或对非正数取对数
               gt_dx = (gt_cx - anchor_cx) / (anchor_w + 1e-8)
               gt_dy = (gt_cy - anchor_cy) / (anchor_h + 1e-8)
               gt_dw = torch.log((gt_w + 1e-8) / (anchor_w + 1e-8))  # 加一个小的 epsilon
               gt_dh = torch.log((gt_h + 1e-8) / (anchor_h + 1e-8))  # 加一个小的 epsilon

               # 设置真实边界框标签为计算出的偏移量 [gt_dx, gt_dy, gt_dw, gt_dh]
               y_batch[anchor_idx, num_classes:] = torch.tensor([gt_dx, gt_dy, gt_dw, gt_dh], device=y_batch.device)
               # --------------------xywh版本----------------------
            # 判断当前 Anchor 是否是负样本
            # 如果一个 Anchor 没有匹配到任何 GT (matched_gt_idx == -1)
            # 或者匹配到的最佳 GT 的 IoU 低于负样本阈值
            # 由于 y_batch 默认是背景，我们只需要确保那些不应该被忽略且 IoU 低于阈值的 Anchor 保持背景标签。
            # 忽略的样本 (IoU 在 negative_iou_threshold 和 positive_iou_threshold 之间) 会保持背景标签，在损失函数中被忽略。
            # 所以这里无需显式处理负样本，默认值已经处理了。

        return y_batch,gt_boxes_padded_norm

def draw_boxes_on_image(image_np, boxes_norm, color=(0, 255, 0), thickness=2, padded_size=(1024, 1024)):
    """
    在图片上绘制边界框。

    Args:
        image_np (numpy.ndarray): 图片的 NumPy 数组，HWC 格式，像素值 0-255。
        boxes_norm (list or numpy.ndarray or torch.Tensor): 边界框列表或数组，
                                                           形状为 (N, 4)，归一化坐标 [xmin, ymin, xmax, ymax] (0-1)。
        color (tuple): 边界框颜色 (B, G, R)。
        thickness (int): 边界框线条粗细。
        padded_size (tuple): 填充后的图片尺寸 (width, height)，用于将归一化坐标转换回像素坐标。
    """
    image_copy = image_np.copy()
    img_h, img_w = image_copy.shape[:2]
    padded_w, padded_h = padded_size

    # 确保输入的 boxes 是 NumPy 数组或 Tensor
    if isinstance(boxes_norm, list):
        boxes_norm = np.array(boxes_norm)
    elif isinstance(boxes_norm, torch.Tensor):
        boxes_norm = boxes_norm.cpu().numpy()

    # 将归一化坐标转换回像素坐标 (相对于 padded_size)
    # 注意：这里假设 image_np 已经是 padded_size 的图像
    boxes_pixel = boxes_norm.copy()
    boxes_pixel[:, 0] = boxes_pixel[:, 0] * padded_w # xmin
    boxes_pixel[:, 1] = boxes_pixel[:, 1] * padded_h # ymin
    boxes_pixel[:, 2] = boxes_pixel[:, 2] * padded_w # xmax
    boxes_pixel[:, 3] = boxes_pixel[:, 3] * padded_h # ymax

    # 确保坐标是整数
    boxes_pixel = boxes_pixel.astype(np.int32)

    # 在图片上绘制每个边界框
    for box in boxes_pixel:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, thickness)

    return image_copy

# 确保 compute_iou_matrix 函数在 test_Dataset 可以访问到
# ... (compute_iou_matrix 函数定义) ...
def test_Dataset():
    # 定义数据增强和预处理操作 (保持与 MyDataset 初始化时一致)
    # 注意：这里我们不应用 transforms.Compose，因为 Dataset.__getitem__ 会自己处理
    # 你需要在 MyDataset 初始化时传入 None 给 transform，并确保内部逻辑正确应用预处理
    # transform = None # 或根据你的实际情况设置
    transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # 调整图片大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(  # 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    # 设置你的数据根目录和参数
    root_dir = param.root_dir # <--- 修改这里为你的数据根目录！
    num_classes = 7 # <--- 根据你的实际类别数设置
    Num_Anchors = param.Num_Anchors # <--- 根据你的计算结果设置
    BACKGROUND_CLASS = 0 # 背景类索引

    # 实例化数据集 (使用训练集部分，以便有真实目标)
    dataset = MyDataset(root_dir=root_dir, train=True, transform=transform, num_classes=num_classes)

    # # --- Debug Print: Check dataset.xml_paths after instantiation ---
    # print("\n--- Debug Print: dataset.xml_paths after instantiation ---")
    # print(f"dataset.xml_paths[:10]: {dataset.xml_list[:min(len(dataset.xml_list), 10)]}")  # 打印前10项
    # print("---------------------------------------------------------")
    # # --- End Debug Print ---

    # 检查 Num_Anchors 是否一致
    if dataset.num_anchors != Num_Anchors:
         print(f"Warning: Dataset reports num_anchors={dataset.num_anchors}, but expected {Num_Anchors}. Check param.py and Dataset.__init__.")

    print("\n--- Testing Dataset and y_batch Generation with Visualization ---")
    # 测试前几张图片的数据加载和 y_batch 生成
    num_images_to_test = min(len(dataset), 5) # 测试最多5张图片

    # --- Debug Print: Check dataset.xml_paths[i] before the loop starts ---
    print("\n--- Debug Print: dataset.xml_paths[i] for test indices before loop ---")
    for i in range(num_images_to_test):
        # 再次打印即将用于测试的索引及其对应的路径
        print(f"  Index {i}: {dataset.xml_list[i]}")
    print("---------------------------------------------")
    # --- End Debug Print ---

    for i in range(num_images_to_test):
        print(f"\nTesting image {i+1}/{num_images_to_test} (Index {dataset.xml_list[i]})") # 打印正在测试的图片对应的 XML 路径
        # 获取图片对应的 XML 文件路径，用于命名保存的图片
        xml_filename = os.path.basename(dataset.xml_list[i])
        image_base_name = os.path.splitext(xml_filename)[0]  # 移除扩展名
        try:
            # 获取数据样本 (假设 __getitem__ 返回 processed_image, y_batch, objects, ...)
            processed_image_tensor, y_batch, objects, original_image, original_size, scale, pad_x, pad_y, padded_size = dataset[i]

            print(f"  processed_image shape: {processed_image_tensor.shape}")
            print(f"  y_batch shape: {y_batch.shape}")
            print(f"  Number of original objects in image: {len(objects)}")

            # --- 检查 y_batch 的形状 ---
            expected_y_batch_shape = torch.Size([Num_Anchors, num_classes + 4])
            assert y_batch.shape == expected_y_batch_shape, f"  y_batch shape mismatch: expected {expected_y_batch_shape}, got {y_batch.shape}"
            print("  y_batch shape is correct.")

            # --- 检查正样本数量 ---
            # 正样本的类别不是背景，且 one-hot 编码中对应类别的概率是 1.0
            # 如果你使用了 One-Hot 编码，正样本行的背景类别应该是 0
            positive_mask_in_ybatch = (y_batch[:, BACKGROUND_CLASS] == 0.0) & (y_batch.max(dim=1)[0] == 1.0) # 排除背景类且是 One-Hot 的

            num_positive_anchors = torch.sum(positive_mask_in_ybatch).item()
            print(f"  Number of positive anchors in y_batch: {num_positive_anchors}")

            # --- 可视化 Anchor Box 和 Ground Truth ---

            # 1. 将 processed_image_tensor 转换为 OpenCV 图像格式 (HWC, BGR, 0-255)
            # 假定 processed_image_tensor 是已经 Normalize 过的
            # 需要反归一化并转换为 HWC 格式
            # Note: 需要使用与 Normalize 相同的均值和方差进行反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # 反归一化
            image_unnormalized = processed_image_tensor * std + mean
            # 将 Tensor 转换为 NumPy 数组，通道顺序从 CHW 变为 HWC
            image_np = image_unnormalized.permute(1, 2, 0).cpu().numpy()
            # 将像素值从 [0, 1] 范围缩放到 [0, 255]，并转换为 uint8 类型
            image_np = (image_np * 255).astype(np.uint8)
            # 转换为 BGR 格式 (OpenCV 默认是 BGR)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


            # 2. 准备 Ground Truth Bounding Boxes (归一化到 1024x1024 填充图的 0-1 范围)
            # 这段逻辑与 convert_to_model_input 中准备 gt_boxes_padded_norm 是一样的
            # 确保这里的计算与 convert_to_model_input 完全一致
            original_gt_boxes_padded_norm = []
            for obj in objects:
                original_bbox = obj['bbox']
            # TODO: 使用与 convert_to_model_input 中相同的逻辑，将 original_bbox 转换为 normalized_bbox
                xmin_padded = original_bbox[0] * scale + pad_x
                ymin_padded = original_bbox[1] * scale + pad_y
                xmax_padded = original_bbox[2] * scale + pad_x
                ymax_padded = original_bbox[3] * scale + pad_y
                # 将边界框裁剪到填充图的像素边界内 (0 到 padded_size-1)
                xmin_padded = max(0, xmin_padded)
                ymin_padded = max(0, ymin_padded)
                xmax_padded = min(padded_size[0], xmax_padded)
                ymax_padded = min(padded_size[1], ymax_padded)

                # 只记录有效框 (宽度和高度 > 0)
                if xmax_padded > xmin_padded and ymax_padded > ymin_padded:
                    normalized_bbox = [
                         xmin_padded / padded_size[0],
                         ymin_padded / padded_size[1],
                         xmax_padded / padded_size[0],
                         ymax_padded / padded_size[1]
                     ]
                     # 确保有效框
                    original_gt_boxes_padded_norm.append(normalized_bbox)
                     # else: 无效框不加入列表
            #3.获取 Reference Boxes (self.reference_boxes 已经在 Dataset.__init__ 中计算并归一化)
            reference_boxes_norm = dataset.reference_boxes  # 形状 (Num_Anchors, 4)，归一化到 0-1 (1024x1024)
            # 4. 在图片上绘制 Ground Truth 和 Anchor Boxes

            # 绘制 Ground Truth (绿色)
            # 即使没有 GT，也要创建图片副本以便绘制 Anchor 或保存
            image_with_gt = draw_boxes_on_image(image_np_bgr.copy(), original_gt_boxes_padded_norm, color=(0, 255, 0),
                                                thickness=2, padded_size=padded_size)

            # 绘制 Reference Boxes (红色)
            # 注意：参考框数量非常多，绘制所有可能会很密集
            # 你可以根据需要调整绘制数量，例如只绘制部分参考框，例如随机选择一些，或者只绘制与 GT 重叠度较高的那些
            # 这里绘制所有 Anchor
            image_with_anchors_and_gt = draw_boxes_on_image(image_with_gt, reference_boxes_norm, color=(0, 0, 255),
                                                            thickness=1, padded_size=padded_size)  # Anchor 用红色，线条细一点

            # --- 保存可视化结果 ---
            output_path_all = os.path.join(visualizations_dir, f"{image_base_name}_gt_anchors.png")
            cv2.imwrite(output_path_all, image_with_anchors_and_gt)
            print(f"  Saved visualization with GT and Anchors to {output_path_all}")

            # 可选：如果你想只保存 GT 框的图片，可以保存 image_with_gt
            # output_path_gt_only = os.path.join(output_dir, f"{image_base_name}_gt_only.png")
            # cv2.imwrite(output_path_gt_only, image_with_gt)
            # print(f"  Saved visualization with GT only to {output_path_gt_only}")
            # --- 打印正样本 Anchor 和对应的真实框信息 (如果存在正样本) ---
            if num_positive_anchors > 0:
                print("  Sample Positive Anchors and their matched Ground Truth:")

                    # 获取所有正样本 Anchor 的索引
                positive_anchor_indices = torch.where(positive_mask_in_ybatch)[0]

                    # 遍历前几个正样本 Anchor (最多打印5个)
                num_samples_to_print = min(num_positive_anchors, 5)

                # 如果没有有效原始真实框，跳过打印对比
                if len(original_gt_boxes_padded_norm) > 0:
                    original_gt_boxes_padded_norm_tensor = torch.tensor(original_gt_boxes_padded_norm, dtype=torch.float32)

                    for k in range(num_samples_to_print):
                        anchor_idx = positive_anchor_indices[k]
                        # 从 y_batch 获取 Anchor 的真实标签和回归目标
                        gt_class_one_hot_in_ybatch = y_batch[anchor_idx, :num_classes]
                        gt_class_idx_in_ybatch = torch.argmax(gt_class_one_hot_in_ybatch).item()
                        gt_box_in_ybatch = y_batch[anchor_idx, num_classes:] # 归一化坐标 (0-1 in padded)  now:xywh

                        # 尝试在原始真实框中找到与 y_batch 中的框匹配的那个
                        # 计算 y_batch 中的框与所有原始真实框的 IoU
                        iou_with_original_gts = compute_iou_matrix(gt_box_in_ybatch.unsqueeze(0), original_gt_boxes_padded_norm_tensor) # 形状 (1, num_valid_original_gts)
                        max_iou_with_original_gts, matched_original_gt_idx = iou_with_original_gts.max(dim=1)

                        if max_iou_with_original_gts.item() > 0.5: # 设定一个高 IoU 阈值来确认匹配
                            matched_original_gt_box = original_gt_boxes_padded_norm_tensor[matched_original_gt_idx.item()]
                            matched_original_gt_class_idx = objects[matched_original_gt_idx.item()]['class_idx'] # 从原始 objects 中获取类别

                            print(f"    Anchor Index: {anchor_idx}")
                            print(f"      y_batch Class: {gt_class_idx_in_ybatch}, Box: [{gt_box_in_ybatch[0]:.4f}, {gt_box_in_ybatch[1]:.4f}, {gt_box_in_ybatch[2]:.4f}, {gt_box_in_ybatch[3]:.4f}]")
                            print(f"      Matched Original GT Class: {matched_original_gt_class_idx}, Box: [{matched_original_gt_box[0]:.4f}, {matched_original_gt_box[1]:.4f}, {matched_original_gt_box[2]:.4f}, {matched_original_gt_box[3]:.4f}]")
                            print(f"      IoU between y_batch box and original GT: {max_iou_with_original_gts.item():.4f}")
                        # else: 可能匹配到了忽略的真实框，或者匹配逻辑有问题，这里不打印

                else:
                     print("  No valid original ground truth boxes in this image to compare with.")


            else:
                print("  No positive anchors found in y_batch for this image.")


        except Exception as e:
            print(f"  Error processing image {i+1}: {e}")
            import traceback
            traceback.print_exc()


    print("\n--- Dataset and y_batch Testing Finished ---")

# -----------------------------------------------------------------------------
# TODO: 确保你的 __getitem__ 方法返回了 objects 列表
# TODO: 确保 compute_iou_matrix 函数在 test_Dataset 可以访问到
# TODO: 仔细检查 test_Dataset 中将 original_bbox 转换为 normalized_bbox 的逻辑，确保与 convert_to_model_input 一致
# -----------------------------------------------------------------------------


def test_Dataset1(dataset_instance, num_samples_to_test=5):
    """
    测试 Dataset 类的数据加载、预处理和 y_batch 生成逻辑。
    包括可视化原始 GT 和 Anchor Box，以及验证 y_batch 中真实偏移量的准确性 (通过解码和 IoU 比较)。

    Args:
        dataset_instance: MyDataset 的一个实例化对象。
        num_samples_to_test: 要测试的图片数量。
    """
    print(f"\n--- Starting Dataset Test for {num_samples_to_test} samples ---")

    # 从 Dataset 实例获取一些全局参数
    num_classes = dataset_instance.num_classes
    background_class = dataset_instance.background_class
    try:
        # 获取 Anchor Boxes (它们是整个 Dataset 共享的)
        # 假设 Dataset 实例有一个 self.reference_boxes 属性，存储了所有的 Anchor Boxes (Num_Anchors, 4) (xyxy, padded_norm)
        all_anchor_boxes_xyxy_padded_norm = torch.tensor(dataset_instance.reference_boxes, dtype=torch.float32)
        num_anchors = all_anchor_boxes_xyxy_padded_norm.shape[0]
        print(f"Loaded {num_anchors} anchor boxes from dataset.")
    except AttributeError:
        print("Error: Dataset instance does not have 'reference_boxes' attribute.")
        print("Cannot perform y_batch decoding verification.")
        all_anchor_boxes_xyxy_padded_norm = None
        num_anchors = 0 # 防止后续使用 num_anchors 报错


    # 创建可视化输出目录
    test_viz_dir = os.path.join(visualizations_dir, "dataset_test_viz")
    os.makedirs(test_viz_dir, exist_ok=True)
    print(f"Saving test visualizations to: {test_viz_dir}")


    for i in range(min(num_samples_to_test, len(dataset_instance))):
        print(f"\nTesting sample {i+1}/{num_samples_to_test}:")
        try:
            # 从 Dataset 获取样本
            # *** IMPORTANT: Adjust this line based on what your MyDataset.__getitem__ actually returns! ***
            # It should return:
            # processed_image_tensor: Processed image tensor (e.g., normalized)
            # y_batch_data: Ground truth tensor (Num_Anchors, num_classes + 4), with [gt_dx, dy, dw, dh] offsets
            # original_image: Original image (e.g., numpy array)
            # original_objects: List of original ground truth objects (from XML)
            # scale, pad_x, pad_y, padded_size: Preprocessing parameters
            # gt_boxes_padded_norm_xyxy_for_test: Original GT boxes in padded_size normalized xyxy format (PyTorch tensor)
            processed_image_tensor, y_batch_data, original_objects, original_image_np,(original_img_width, original_img_height),scale, pad_x, pad_y, padded_size, gt_boxes_padded_norm_xyxy_for_test = dataset_instance[i]

            print(f"  Processed image shape: {processed_image_tensor.shape}")
            print(f"  y_batch shape: {y_batch_data.shape}")
            print(f"  Number of original objects in image: {len(original_objects)}")
            print(f"  Padded image size: {padded_size}")
            print(f"  Scale: {scale:.4f}, Pad_x: {pad_x}, Pad_y: {pad_y}")

            # --- Check y_batch shape ---
            if num_anchors > 0: # Only check if anchor_boxes were loaded
                expected_y_batch_shape = torch.Size([num_anchors, num_classes + 4])
                assert y_batch_data.shape == expected_y_batch_shape, f"  y_batch shape mismatch: expected {expected_y_batch_shape}, got {y_batch_data.shape}"
                print("  y_batch shape is correct.")

            # --- Extract data from y_batch ---
            gt_class_one_hot_in_ybatch = y_batch_data[:, :num_classes]
            gt_bbox_offsets_in_ybatch = y_batch_data[:, num_classes:] # [gt_dx, gt_dy, gt_dw, gt_dh]

            gt_class_indices_in_ybatch = torch.argmax(gt_class_one_hot_in_ybatch, dim=1)

            # Filter out background/ignored anchors to find positive ones
            # Positive anchors are those not marked as background and with a positive class one-hot value
            positive_anchor_mask = (gt_class_indices_in_ybatch != background_class) & \
                                   (gt_class_one_hot_in_ybatch.max(dim=1)[0] == 1.0) # Check if it's a valid foreground one-hot
            positive_anchor_indices = torch.where(positive_anchor_mask)[0]

            print(f"  Number of positive anchors in y_batch: {len(positive_anchor_indices)}")

            # --- Visualize Ground Truth and Anchor Boxes ---

            # 1. Convert processed_image_tensor back to OpenCV format (HWC, BGR, 0-255)
            # Assuming it was normalized with mean/std
            try:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_unnormalized = processed_image_tensor.cpu() * std + mean
                image_np = image_unnormalized.permute(1, 2, 0).numpy()
                image_np = (image_np * 255).astype(np.uint8)
                image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            except Exception as e:
                 print(f"  Warning: Could not convert processed_image_tensor to OpenCV format for visualization: {e}")
                 image_np_bgr = np.zeros((padded_size[1], padded_size[0], 3), dtype=np.uint8) # Create a black image placeholder


            # 2. Draw Ground Truth (from gt_boxes_padded_norm_xyxy_for_test) (Green)
            # gt_boxes_padded_norm_xyxy_for_test is a (M, 4) PyTorch tensor (padded_norm xyxy)
            if gt_boxes_padded_norm_xyxy_for_test is not None and gt_boxes_padded_norm_xyxy_for_test.shape[0] > 0:
                 gt_boxes_padded_norm_list = gt_boxes_padded_norm_xyxy_for_test.tolist()
                 image_with_gt = draw_boxes_on_image(image_np_bgr.copy(), gt_boxes_padded_norm_list, color=(0, 255, 0),
                                                     thickness=2, padded_size=padded_size)
            else:
                 image_with_gt = image_np_bgr.copy() # No GT to draw, just copy the image
                 print("  No original ground truth boxes in padded_norm for visualization.")


            # # 3. Draw Reference Boxes (Red)
            # if all_anchor_boxes_xyxy_padded_norm is not None:
            #      reference_boxes_list = all_anchor_boxes_xyxy_padded_norm.tolist()
            #      # Note: Drawing all anchors can be very dense. You might want to sample a subset.
            #      # For this test, we draw all.
            #      image_with_anchors_and_gt = draw_boxes_on_image(image_with_gt, reference_boxes_list, color=(0, 0, 255),
            #                                                      thickness=1, padded_size=padded_size)
            # else:
            #      image_with_anchors_and_gt = image_with_gt.copy() # No anchors to draw

            # 3. Draw Matched Positive Anchor Boxes (e.g., Blue)
            if len(positive_anchor_indices) > 0 and all_anchor_boxes_xyxy_padded_norm is not None:
                     # Select the coordinates of the positive anchor boxes
                matched_anchor_boxes_xyxy_padded_norm = all_anchor_boxes_xyxy_padded_norm[positive_anchor_indices]
                matched_anchor_boxes_list = matched_anchor_boxes_xyxy_padded_norm.tolist()

                     # Draw matched positive anchor boxes on the current image
                image_with_gt = draw_boxes_on_image(image_with_gt,matched_anchor_boxes_list, color=(255, 0, 0),thickness=1, padded_size=padded_size)
                print(f"  Drawn {len(matched_anchor_boxes_list)} Matched Positive Anchor boxes (Blue).")

            elif len(positive_anchor_indices) > 0 and all_anchor_boxes_xyxy_padded_norm is None:
                print("  Cannot visualize matched positive anchors because anchor boxes are not available.")
            else:
                print("  No positive anchors found to visualize.")

            # Save visualization
            image_base_name = f"sample_{i}_viz" # Generic name if XML path is not available
            try:
                 xml_filename = os.path.basename(dataset_instance.xml_list[i]) # Assuming xml_list is an attribute
                 image_base_name = os.path.splitext(xml_filename)[0]
            except AttributeError:
                 pass # Use generic name


            output_path_all = os.path.join(test_viz_dir, f"{image_base_name}_gt_anchors.png")
            cv2.imwrite(output_path_all, image_with_gt)# image_with_anchors_and_gt
            print(f"  Saved visualization with GT and Anchors to {output_path_all}")


            # --- Verify Positive Anchors (Decode y_batch and Compare with Original GT) ---
            if len(positive_anchor_indices) > 0 and all_anchor_boxes_xyxy_padded_norm is not None:
                print(f"  Verifying {len(positive_anchor_indices)} positive anchors:")

                # Ensure original GT boxes are available for comparison
                if gt_boxes_padded_norm_xyxy_for_test is not None and gt_boxes_padded_norm_xyxy_for_test.shape[0] > 0:

                    # Iterate through positive anchors and perform verification
                    for anchor_idx in positive_anchor_indices:
                        gt_class_idx = gt_class_indices_in_ybatch[anchor_idx].item()
                        gt_bbox_offsets = gt_bbox_offsets_in_ybatch[anchor_idx].unsqueeze(0).to('cpu') # Move to CPU for decoding
                        corresponding_anchor_box = all_anchor_boxes_xyxy_padded_norm[anchor_idx].unsqueeze(0).to('cpu') # Move to CPU

                        # Decode the ground truth offsets from y_batch   解码!!!!!1
                        decoded_gt_box_padded_norm_from_ybatch = decode_boxes(
                            corresponding_anchor_box,
                            gt_bbox_offsets
                        ).squeeze(0) # Shape (4,) [xmin, ymin, xmax, ymax] padded_norm

                        # Check if the decoded box is valid
                        if decoded_gt_box_padded_norm_from_ybatch[2] <= decoded_gt_box_padded_norm_from_ybatch[0] or \
                           decoded_gt_box_padded_norm_from_ybatch[3] <= decoded_gt_box_padded_norm_from_ybatch[1]:
                             print(f"    Anchor Index {anchor_idx} (Class {gt_class_idx}): Warning: Decoded y_batch box is invalid: {decoded_gt_box_padded_norm_from_ybatch.tolist()}. Skipping IoU check.")
                             continue # Skip IoU check for invalid box

                        # Compute IoU with all original GT boxes for this image
                        iou_matrix_with_original_gts = compute_iou_matrix(
                            decoded_gt_box_padded_norm_from_ybatch.unsqueeze(0), # Decoded box (1, 4)
                            gt_boxes_padded_norm_xyxy_for_test.to('cpu') # Original GTs (M, 4) - move to CPU
                        )

                        # Find the best matching original GT and its IoU
                        max_iou_with_original_gts, matched_original_gt_idx = torch.max(iou_matrix_with_original_gts, dim=1)
                        max_iou = max_iou_with_original_gts.item()

                        print(f"    Anchor Index {anchor_idx} (Class {gt_class_idx}): IoU between decoded y_batch box and best matching Original GT: {max_iou:.4f}")

                        # Optional: Print details of the best matching original GT if IoU is high
                        if max_iou > 0.5: # Set a threshold for printing details
                            matched_original_gt_padded_norm_xyxy = gt_boxes_padded_norm_xyxy_for_test[matched_original_gt_idx.item()].tolist()
                            print(f"      Best Matching Original GT Box (padded_norm xyxy): {matched_original_gt_padded_norm_xyxy}")


                else:
                     print("  No original ground truth boxes in padded_norm available for y_batch verification.")

            elif len(positive_anchor_indices) > 0 and all_anchor_boxes_xyxy_padded_norm is None:
                 print("  Cannot verify y_batch for positive anchors because anchor boxes are not available.")

            else:
                print("  No positive anchors found in y_batch to verify.")


        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()


    print("\n--- Dataset Test Finished ---")
# 在脚本的 main 部分调用 test_Dataset() 来运行测试
if __name__ == "__main__":
    # 确保你的 MyDataset 类已经定义好了
    # 实例化数据集 (使用训练集部分，以便有真实目标)
    root_dir = param.root_dir # <--- 修改这里为你的数据根目录！
    num_classes = 7 # <--- 根据你的实际类别数设置
    Num_Anchors = param.Num_Anchors # <--- 根据你的计算结果设置
    BACKGROUND_CLASS = 0 # 背景类索引
    # 定义数据增强和预处理操作 (保持与 MyDataset 初始化时一致)
    # 这里只定义 ToTensor 和 Normalize，因为 MyDataset.__getitem__ 自己处理缩放填充等
    transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(  # 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    # 实例化数据集 (使用训练集部分，以便有真实目标)
    # *** IMPORTANT: Ensure MyDataset.__init__ calculates and stores self.reference_boxes ***
    # *** IMPORTANT: Ensure MyDataset.__getitem__ returns ALL the necessary data as listed in the test function ***
    try:
        dataset_instance = MyDataset(root_dir=root_dir, train=True, transform=transform, num_classes=num_classes)
        # 调用测试函数
        test_Dataset1(dataset_instance, num_samples_to_test=5)
    except Exception as e:
        print(f"\nError creating Dataset instance or running test: {e}")
        import traceback
        traceback.print_exc()