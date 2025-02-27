import xml.etree.ElementTree as ET
import torch
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
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

def convert_to_model_input(objects, num_anchors, num_classes):
    # 初始化标签张量
    y_batch = torch.zeros((1, num_anchors, num_classes + 4))  # batch_size=1

    # 填充标签
    for i, obj in enumerate(objects):
        if i >= num_anchors:
            break  # 超过锚框数量，停止填充
        name = obj['name']
        bbox = obj['bbox']

        # 设置分类标签
        y_batch[0, i, name_to_index[name]] = 1.0  # 假设 name_to_index 是类别名称到索引的映射

        # 设置边界框标签
        y_batch[0, i, num_classes:] = torch.tensor(bbox)  # [xmin, ymin, xmax, ymax]

    return y_batch

name_to_index = {
    'missing_hole': 0,
    'mouse_bite':1,
    'Open_circuit':2,
    'Short':3,
    'Spur':4,
    'Spurious_copper':5
    # 添加其他类别
}