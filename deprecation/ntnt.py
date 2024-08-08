import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入灰度图像
# image = cv2.imread('area2_2.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('../img.png', cv2.IMREAD_GRAYSCALE)

# 检测边缘
edges = cv2.Canny(image, 700, 400)

# 找到边缘的轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 形态学 Top Hat 变换
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
top_hat = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)

# 二值化
_, binary_image = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 在二值化图像上将轮廓点设为黑色
cv2.drawContours(binary_image, contours, -1, (0), thickness=100)

# 形态学腐蚀，获得整体切屑轮廓
kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(binary_image, kernel_erode, iterations=1)

# 形态学膨胀，获得整个切屑特征
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)

# 显示结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 5, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 5, 2)
plt.title('Edges Detected')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 5, 3)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')

plt.subplot(2, 5, 4)
plt.title('Top Hat')
plt.imshow(top_hat, cmap='gray')

plt.subplot(2, 5, 5)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 5, 6)
plt.title('Eroded and Dilated Image')
plt.imshow(dilated, cmap='gray')

cv2.imwrite('../b.png', binary_image)

plt.tight_layout()
plt.show()
