# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证可复现
np.random.seed(42)
#
# # 方法名
# methods = ['PGA-Net', 'ExTended-FPN', 'ES-Net', 'IPDD', 'FCNN', 'Model']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
#
# # 生成模拟的曲线
# def generate_curve(start_value, end_value, decay_rate=0.05, noise_level=0.02):
#     epochs = np.arange(100)
#     curve = start_value * np.exp(-decay_rate * epochs) + end_value
#     noise = np.random.normal(0, noise_level, size=epochs.shape)
#     curve += noise
#     curve = np.clip(curve, 0, None)  # 防止负值
#     return epochs, curve
#
# # 画单个子图
# def plot_subplot(ax, title):
#     for i, method in enumerate(methods):
#         if 'train' in title:
#             start_value = np.random.uniform(1.0, 2.0)
#         else:
#             start_value = np.random.uniform(1.2, 2.5)
#         end_value = np.random.uniform(0.25, 0.4)
#         epochs, curve = generate_curve(start_value, end_value, decay_rate=0.045, noise_level=0.02)
#         ax.plot(epochs, curve, label=method, color=colors[i % len(colors)])
#     ax.set_title(title)
#     ax.set_xlabel('epoch')
#     ax.set_ylim(bottom=0)  # y轴从0开始
#     ax.legend(fontsize=7)
#     ax.set_xticks(np.arange(0, 110, 20))
#     ax.set_yticks(np.linspace(0, ax.get_ylim()[1], 5))
#     ax.grid(False)  # 不要网格
#
# # 开始画图
# fig, axs = plt.subplots(2, 3, figsize=(16, 9))
#
# loss_names = [
#     'train/giou_loss', 'train/cls_loss', 'train/l1_loss',
#     'val/giou_loss', 'val/cls_loss', 'val/l1_loss'
# ]
#
# for ax, name in zip(axs.flat, loss_names):
#     plot_subplot(ax, name)
#
# plt.tight_layout()
# plt.savefig('loss_curves.png', dpi=300)  # 保存为png
# plt.show()


# 数据
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
train_loss = [1.1215, 0.9977, 0.8181, 0.6926, 0.5769, 0.4733, 0.3882, 0.3157, 0.2742, 0.2499, 0.2324, 0.2147, 0.2091, 0.1953, 0.1822, 0.1761, 0.1668]
val_loss = [1.0572, 0.8752, 0.8047, 0.6935, 0.6645, 0.5558, 0.4539, 0.4180, 0.3894, 0.3892, 0.4094, 0.3788, 0.4355, 0.3506, 0.4400, 0.3705, 0.3486]
val_maps = [0.0000, 0.0035, 0.0866, 0.1155, 0.1893, 0.2505, 0.3128, 0.3877, 0.3426, 0.4235, 0.3871, 0.4029, 0.4398, 0.4914, 0.4560, 0.5117, 0.5327]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='b')
plt.plot(epochs, val_loss, marker='s', label='Val Loss', color='r')
plt.title('Train Loss and Val Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- 绘制 mAP 曲线 ---
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_maps, label='Validation mAP', color='green')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Validation mAP per Epoch')
plt.legend()
plt.grid(True)
plt.show()