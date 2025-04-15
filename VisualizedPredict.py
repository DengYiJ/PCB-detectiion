import os
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def nms(boxes, scores, iou_threshold=0.5):
    # �����Ŷ�����
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        # �����÷���ߵĿ�
        keep.append(order[0])

        # ����IoU
        ious = calculate_iou(boxes[order[0]], boxes[order[1:]])

        # �Ƴ�IoU������ֵ�Ŀ�
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
def visualize_predictions(original_images, y_pre, y_batch, output_dir, epoch, num_classes, original_sizes):
    # ������ͼ���ϻ���Ԥ���ê��ͷ�����Ϣ������������浽���ļ�����
    # ȷ�����Ŀ¼����
    os.makedirs(output_dir, exist_ok=True)
    # �����������
    # class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
 # 定义类别名称（跳过背景类0）
    class_names = {
        1: 'missing_hole',
        2: 'mouse_bite', 
        3: 'open_circuit',
        4: 'short',
        5: 'spur',
        6: 'spurious_copper'
    }
    # ���ͼ��������������ת��Ϊ numpy ����
    # if isinstance(images, torch.Tensor):
    #     images = images.cpu().numpy()
    #     images = np.transpose(images, (0, 2, 3, 1))  # �� (B, C, H, W) ת��Ϊ (B, H, W, C)
    #     images = (images * 255).astype(np.uint8)
    # else:
    #     # ���ͼ���� numpy ���飬ȷ������״������������ȷ
    #     images = np.array(images)
    #     if images.dtype != np.uint8:
    #         images = (images * 255).astype(np.uint8)

    # ��Ԥ��ͱ�ǩ������ת��Ϊ numpy ����
    if isinstance(y_pre, torch.Tensor):
        y_pre = y_pre.cpu().numpy()
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.cpu().numpy()

    for i in range(len(original_images)):
        # ����ͼ�񸱱�
        image = original_images[i].copy()
        orig_height, orig_width = image.shape[:2]

        # ����Ԥ���
        for anchor in range(y_pre[i].shape[0]):
            class_probs = y_pre[i, anchor, :num_classes]
            bbox = y_pre[i, anchor, num_classes:]
            class_idx = np.argmax(class_probs)
            score = class_probs[class_idx]

            if class_idx == 0 or score<=0.5:
                continue

            if score > 0.5:  # ���Ŷ���ֵ
                # ��ȡ��һ���ı߽������
                xmin, ymin, xmax, ymax = bbox

                # �޸�����ת������
                scale_x = orig_width / 1024.0
                scale_y = orig_height / 1024.0

                # �ȷ���һ����1024x1024����
                xmin = bbox[0] * 1024
                ymin = bbox[1] * 1024
                xmax = bbox[2] * 1024
                ymax = bbox[3] * 1024

                # ����һ����ԭʼͼ��ߴ�
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_x)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_x)

                # ȷ��������ͼ��Χ��
                xmin = max(0, min(xmin, orig_width))
                ymin = max(0, min(ymin, orig_height))
                xmax = max(0, min(xmax, orig_width))
                ymax = max(0, min(ymax, orig_height))

                # ����Ԥ�����ɫ��
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"{class_names[class_idx]}: {score:.2f}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ������ʵ��
        for anchor in range(y_batch[i].shape[0]):
            class_probs = y_batch[i, anchor, :num_classes]
            bbox = y_batch[i, anchor, num_classes:]
            class_idx = np.argmax(class_probs)

             # 跳过背景类(class_idx=0)和无效标注
            if class_idx == 0 or class_probs[class_idx] <= 0:
                continue
                        # 获取类别名称（确保在有效范围内）
            class_name = class_names.get(class_idx, f"unknown_{class_idx}")
            
            if class_probs[class_idx] > 0:  # ֻ������Ч����ʵ��
                # ����һ����ԭʼͼ��ߴ�
                xmin, ymin, xmax, ymax = bbox
                xmin = int(xmin * orig_width)
                ymin = int(ymin * orig_height)
                xmax = int(xmax * orig_width)
                ymax = int(ymax * orig_height)

                # ȷ��������ͼ��Χ��
                xmin = max(0, min(xmin, orig_width))
                ymin = max(0, min(ymin, orig_height))
                xmax = max(0, min(xmax, orig_width))
                ymax = max(0, min(ymax, orig_height))

                # ������ʵ����ɫ��
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(image, f"{class_names[class_idx]}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ������
        output_path = os.path.join(output_dir, f"epoch_{epoch}_image_{i}.jpg")
        # ���ͼ����RGB��ʽ����Ҫת��ΪBGR��ʽ�ٱ���
        if image.shape[-1] == 3:  # ����Ƿ��ǲ�ɫͼ��
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)