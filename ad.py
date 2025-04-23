import copy
import os
import random
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

import cv2
import numpy as np


def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    class_names = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        name = obj.find("name").text
        annotations.append([xmin, ymin, xmax, ymax])
        class_names.append(name)

    return annotations, class_names


def save_updated_xml(xml_path, image_name, img_width, img_height, bboxes, class_names):
    root = Element("annotation")
    SubElement(root, "filename").text = image_name
    size = SubElement(root, "size")
    SubElement(size, "width").text = str(img_width)
    SubElement(size, "height").text = str(img_height)
    SubElement(size, "depth").text = "3"

    for bbox, class_name in zip(bboxes, class_names):
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = class_name
        bndbox = SubElement(obj, "bndbox")
        SubElement(bndbox, "xmin").text = str(bbox[0])
        SubElement(bndbox, "ymin").text = str(bbox[1])
        SubElement(bndbox, "xmax").text = str(bbox[2])
        SubElement(bndbox, "ymax").text = str(bbox[3])

    tree = ElementTree(root)
    tree.write(xml_path)


def random_flip(image, bboxes, p_hflip=0.5, p_vflip=0.1):
    h, w, _ = image.shape
    new_bboxes = copy.deepcopy(bboxes)
    if random.random() < p_hflip:
        image = cv2.flip(image, 1)
        new_bboxes = [[w - xmax, ymin, w - xmin, ymax] for xmin, ymin, xmax, ymax in new_bboxes]
    if random.random() < p_vflip:
        image = cv2.flip(image, 0)
        new_bboxes = [[xmin, h - ymax, xmax, h - ymin] for xmin, ymin, xmax, ymax in new_bboxes]

    return image, new_bboxes


def random_brightness_contrast(image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)):
    brightness_factor = random.uniform(*brightness_range)
    contrast_factor = random.uniform(*contrast_range)
    image = image.astype(np.float32)
    image = image * brightness_factor
    image = image + (image.mean() * (contrast_factor - 1))
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_hsv(image, hue_range=(-0.05, 0.05), sat_range=(0.5, 1.5), val_range=(0.5, 1.5)):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(image_hsv)

    h = h + random.uniform(*hue_range) * 360
    s = s * random.uniform(*sat_range)
    v = v * random.uniform(*val_range)

    image_hsv = cv2.merge([h, s, v])
    image_hsv = np.clip(image_hsv, 0, 255)
    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_rotation(image, bboxes, angle_range=(-10, 10)):
    h, w, _ = image.shape
    angle = random.uniform(*angle_range)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(0, 0, 0))

    new_bboxes = []
    for bbox in bboxes:
        points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                           [bbox[2], bbox[3]], [bbox[0], bbox[3]]], dtype=np.float32)
        rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]
        x_coords, y_coords = rotated_points[:, 0], rotated_points[:, 1]
        new_bboxes.append([int(min(x_coords)), int(min(y_coords)),
                           int(max(x_coords)), int(max(y_coords))])

    valid_bboxes = []
    for bbox in new_bboxes:
        xmin, ymin, xmax, ymax = bbox
        if xmin < xmax and ymin < ymax and 0 <= xmin < w and 0 <= ymin < h:
            valid_bboxes.append(bbox)
    return rotated_image, valid_bboxes


def random_scale(image, bboxes, scale_range=(0.8, 1.2)):
    h, w, _ = image.shape
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)

    scaled_image = cv2.resize(image, (new_w, new_h))

    new_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = [int(coord * scale) for coord in bbox]
        new_bboxes.append([xmin, ymin, xmax, ymax])

    return scaled_image, new_bboxes


def random_gaussian_blur(image, kernel_range=(3, 7)):
    kernel_size = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def random_cutout(image, bboxes, max_holes=3, max_size=24, p=0.3):
    if random.random() > p:
        return image, bboxes

    h, w, _ = image.shape
    new_bboxes = copy.deepcopy(bboxes)

    for _ in range(random.randint(1, max_holes)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        size = random.randint(8, max_size)

        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(w, x + size // 2), min(h, y + size // 2)

        image[y1:y2, x1:x2] = 0

        valid_bboxes = []
        for bbox in new_bboxes:
            bx1, by1, bx2, by2 = bbox
            if not (bx2 <= x1 or bx1 >= x2 or by2 <= y1 or by1 >= y2):
                valid_bboxes.append(bbox)
        new_bboxes = valid_bboxes

    return image, new_bboxes


def apply_augmentations(image, bboxes, class_names):
    bboxes = copy.deepcopy(bboxes)
    class_names = copy.deepcopy(class_names)

    image, bboxes = random_flip(image, bboxes, p_hflip=0.5, p_vflip=0.1)

    if random.random() < 0.5:
        image, bboxes = random_scale(image, bboxes, scale_range=(0.8, 1.2))

    if random.random() < 0.3:
        image, bboxes = random_rotation(image, bboxes, angle_range=(-10, 10))

    if random.random() < 0.5:
        image = random_brightness_contrast(image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5))

    if random.random() < 0.5:
        image = random_hsv(image, hue_range=(-0.05, 0.05), sat_range=(0.5, 1.5), val_range=(0.5, 1.5))

    if random.random() < 0.3:
        image = random_gaussian_blur(image, kernel_range=(3, 7))

    if random.random() < 0.3:
        image, bboxes = random_cutout(image, bboxes, max_holes=3, max_size=24, p=0.3)

    h, w, _ = image.shape
    valid_bboxes = []
    valid_class_names = []
    for bbox, class_name in zip(bboxes, class_names):
        xmin, ymin, xmax, ymax = bbox
        if xmin < xmax and ymin < ymax and 0 <= xmin < w and 0 <= ymin < h:
            valid_bboxes.append(bbox)
            valid_class_names.append(class_name)

    return image, valid_bboxes, valid_class_names


def augment_dataset(image_dir, xml_dir, output_dir, augment_count, max_attempts=10):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        xml_path = os.path.join(xml_dir, image_file.replace(".jpg", ".xml"))

        if not os.path.exists(xml_path):
            print(f"Warning: No XML annotation found for {image_file}. Skipping...")
            continue

        image = cv2.imread(image_path)
        bboxes, class_names = parse_voc_xml(xml_path)
        base_name = image_file.replace(".jpg", "")
        cv2.imwrite(os.path.join(output_dir, "images", image_file), image)
        save_updated_xml(
            os.path.join(output_dir, "annotations", image_file.replace(".jpg", ".xml")),
            image_file, image.shape[1], image.shape[0], bboxes, class_names
        )

        for i in range(augment_count):
            attempts = 0
            while attempts < max_attempts:
                augmented_image, augmented_bboxes, augmented_class_names = apply_augmentations(
                    image.copy(), bboxes.copy(), class_names.copy()
                )

                if augmented_bboxes:
                    augmented_image_name = f"{base_name}_aug_{i + 1}.jpg"
                    augmented_image_path = os.path.join(output_dir, "images", augmented_image_name)
                    cv2.imwrite(augmented_image_path, augmented_image)

                    augmented_xml_path = os.path.join(output_dir, "annotations",
                                                      augmented_image_name.replace(".jpg", ".xml"))
                    save_updated_xml(augmented_xml_path, augmented_image_name,
                                     augmented_image.shape[1], augmented_image.shape[0],
                                     augmented_bboxes, augmented_class_names)
                    print(f"Success: {augmented_image_name} saved with {len(augmented_bboxes)} bboxes")
                    break
                else:
                    attempts += 1
                    print(f"Attempt {attempts} failed for {image_file}_aug_{i + 1}: No valid bboxes")

            if attempts >= max_attempts:
                print(f"Warning: Skipped {image_file}_aug_{i + 1} after {max_attempts} attempts: No valid bboxes")


augment_dataset(
    image_dir=r"D:\PycharmProjects\PCB\data\images",
    xml_dir=r"D:\PycharmProjects\PCB\data\Annotations",
    output_dir=r"D:\PycharmProjects\PCB\data\output",
    augment_count=6,
    max_attempts=10
)
