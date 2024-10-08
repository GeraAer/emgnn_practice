import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于科学计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import os  # 导入os库，用于处理文件和目录

# 设置图像序列所在的文件夹路径
image_folder = 'images'  # 请确保此文件夹中包含你下载的图像

# 获取图像文件列表
image_files = sorted([os.path.join(image_folder, f"{i}.jpg") for i in range(1, 11)])

# 创建背景减除器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 读取图像序列并进行前景检测
frames = []
foreground_masks = []
for image_file in image_files:
    frame = cv2.imread(image_file)
    frames.append(frame)
    fg_mask = bg_subtractor.apply(frame)
    foreground_masks.append(fg_mask)

# 获取背景图像
background_image = bg_subtractor.getBackgroundImage()

# 展示图像序列
plt.figure(figsize=(15, 10))
for i, frame in enumerate(frames):
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Frame {i + 1}')
    plt.axis('off')
plt.suptitle('Image Sequence')
plt.show()

# 展示背景建模的结果
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
plt.title('Background Image')
plt.axis('off')
plt.show()

# 展示前景检测结果
plt.figure(figsize=(15, 10))
for i, fg_mask in enumerate(foreground_masks):
    plt.subplot(2, 5, i + 1)
    plt.imshow(fg_mask, cmap='gray')
    plt.title(f'Foreground Mask {i + 1}')
    plt.axis('off')
plt.suptitle('Foreground Detection Results')
plt.show()
