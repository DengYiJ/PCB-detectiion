# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import math
import random
from PIL import Image
from param import input_annotations_dir,input_images_dir


def rotate_point(x, y, cx, cy, angle_rad):
    # 相对于中心点的旋转
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    x_rel = x - cx
    y_rel = y - cy

    rotated_x = cx + (x_rel * cos_theta - y_rel * sin_theta)
    rotated_y = cy + (x_rel * sin_theta + y_rel * cos_theta)

    return rotated_x, rotated_y


def rotate_image_and_xml(image_path, xml_path, output_image_path, output_xml_path, angle):
    # 打开图像并旋转
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)
    rotated_image.save(output_image_path)

    # 解析原始XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取原始图像的宽度和高度
    original_width = int(root.find('size/width').text)
    original_height = int(root.find('size/height').text)

    # 计算旋转后的图像尺寸
    angle_rad = math.radians(angle)
    new_width = int(abs(original_width * math.cos(angle_rad)) + abs(original_height * math.sin(angle_rad)))
    new_height = int(abs(original_height * math.cos(angle_rad)) + abs(original_width * math.sin(angle_rad)))

    # 更新图像尺寸信息
    root.find('size/width').text = str(new_width)
    root.find('size/height').text = str(new_height)

    # 计算旋转中心
    # center_x = original_width / 2
    # center_y = original_height / 2
    #
    # # 遍历每个目标并更新坐标
    # for obj in root.findall('object'):
    #     bndbox = obj.find('bndbox')
    #     xmin = int(bndbox.find('xmin').text)
    #     ymin = int(bndbox.find('ymin').text)
    #     xmax = int(bndbox.find('xmax').text)
    #     ymax = int(bndbox.find('ymax').text)
    #
    #     # 将坐标转换为相对于中心点的坐标
    #     x1 = xmin - center_x
    #     y1 = ymin - center_y
    #     x2 = xmax - center_x
    #     y2 = ymax - center_y
    #
    #     # 旋转坐标
    #     rotated_x1 = x1 * math.cos(angle_rad) - y1 * math.sin(angle_rad)
    #     rotated_y1 = x1 * math.sin(angle_rad) + y1 * math.cos(angle_rad)
    #     rotated_x2 = x2 * math.cos(angle_rad) - y2 * math.sin(angle_rad)
    #     rotated_y2 = x2 * math.sin(angle_rad) + y2 * math.cos(angle_rad)
    #
    #     # 将坐标转换回原始坐标系
    #     rotated_xmin = rotated_x1 + new_width / 2
    #     rotated_ymin = rotated_y1 + new_height / 2
    #     rotated_xmax = rotated_x2 + new_width / 2
    #     rotated_ymax = rotated_y2 + new_height / 2
    #
    #     # 更新XML中的坐标
    #     bndbox.find('xmin').text = str(int(round(rotated_xmin)))
    #     bndbox.find('ymin').text = str(int(round(rotated_ymin)))
    #     bndbox.find('xmax').text = str(int(round(rotated_xmax)))
    #     bndbox.find('ymax').text = str(int(round(rotated_ymax)))
    # 计算旋转中心
    center_x = original_width / 2
    center_y = original_height / 2
    angle_rad = math.radians(angle)
    # 遍历每个目标并更新坐标
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 将坐标转换为相对于中心点的坐标
        x1 = xmin - center_x
        y1 = ymin - center_y
        x2 = xmax - center_x
        y2 = ymax - center_y

        # 旋转坐标
        rotated_x1 = x1 * math.cos(angle_rad) - y1 * math.sin(angle_rad)
        rotated_y1 = x1 * math.sin(angle_rad) + y1 * math.cos(angle_rad)
        rotated_x2 = x2 * math.cos(angle_rad) - y2 * math.sin(angle_rad)
        rotated_y2 = x2 * math.sin(angle_rad) + y2 * math.cos(angle_rad)

        # 将坐标转换回原始坐标系
        rotated_xmin = rotated_x1 + new_width / 2
        rotated_ymin = rotated_y1 + new_height / 2
        rotated_xmax = rotated_x2 + new_width / 2
        rotated_ymax = rotated_y2 + new_height / 2

        # 更新XML中的坐标
        bndbox.find('xmin').text = str(int(round(rotated_xmin)))
        bndbox.find('ymin').text = str(int(round(rotated_ymin)))
        bndbox.find('xmax').text = str(int(round(rotated_xmax)))
        bndbox.find('ymax').text = str(int(round(rotated_ymax)))

    # 保存更新后的XML文件
    tree.write(output_xml_path)

def preprocess_dataset(input_images_dir, input_annotations_dir, output_images_dir, output_annotations_dir, angle_range=(-10, 10)):
    # 确保输出目录存在
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # 遍历所有类别
    for category in os.listdir(input_images_dir):
        input_category_images_dir = os.path.join(input_images_dir, category)
        input_category_annotations_dir = os.path.join(input_annotations_dir, category)
        output_category_images_dir = os.path.join(output_images_dir, category)
        output_category_annotations_dir = os.path.join(output_annotations_dir, category)

        # 确保输出类别目录存在
        os.makedirs(output_category_images_dir, exist_ok=True)
        os.makedirs(output_category_annotations_dir, exist_ok=True)

        # 遍历所有图像
        for image_file in os.listdir(input_category_images_dir):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(input_category_images_dir, image_file)
                xml_file = os.path.splitext(image_file)[0] + '.xml'
                xml_path = os.path.join(input_category_annotations_dir, xml_file)

                # 生成随机旋转角度
                angle = random.uniform(angle_range[0], angle_range[1])

                # 生成输出文件名
                base_name = os.path.splitext(image_file)[0]
                output_image_file = f"{base_name}_rotated_{angle:.2f}.jpg"
                output_xml_file = f"{base_name}_rotated_{angle:.2f}.xml"

                # 生成输出路径
                output_image_path = os.path.join(output_category_images_dir, output_image_file)
                output_xml_path = os.path.join(output_category_annotations_dir, output_xml_file)

                # 旋转图像并更新XML
                rotate_image_and_xml(image_path, xml_path, output_image_path, output_xml_path, angle)

                print(f"Processed {image_file} with angle {angle:.2f}")

if __name__ == "__main__":
    # 输入和输出目录
    input_images_dir = input_images_dir
    input_annotations_dir = input_annotations_dir
    output_images_dir = "./PCB_DATASET/rotated_images"
    output_annotations_dir = "./PCB_DATASET/rotated_annotations"

    # 角度范围
    angle_range = (-10, 10)

    # 运行预处理
    preprocess_dataset(input_images_dir, input_annotations_dir, output_images_dir, output_annotations_dir, angle_range)