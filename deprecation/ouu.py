import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀入灰度圖像
image = cv2.imread('../test.png', cv2.IMREAD_GRAYSCALE)

# 去噪
# image = cv2.GaussianBlur(image, (5, 5), 0)

# 邊緣檢測
edges = cv2.Canny(image, 350, 1000)

# 找到邊緣的輪廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 創建一個與圖像大小相同的黑色蒙版
mask = np.zeros_like(image)

# 在蒙版上填充所有外部輪廓
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# 在原圖上將輪廓點設為黑色
image_with_black_contours = image.copy()
# cv2.drawContours(image_with_black_contours, contours, -1, (0), thickness=1000)

# 反轉蒙版
mask = cv2.bitwise_not(mask)

# 使用蒙版去除邊緣
image_without_edges = cv2.bitwise_and(image, image, mask=mask)

# 自適應直方圖均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(image_without_edges)

# 形態學 Top Hat 和 Bottom Hat 變換
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
top_hat = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)
# bottom_hat = cv2.morphologyEx(equalized_image, cv2.MORPH_BLACKHAT, kernel)
# combined = cv2.add(top_hat, bottom_hat)

# 二值化
_, binary_image = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 形態學操作
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open, iterations=2)

# 顯示結果
plt.figure(figsize=(12, 8))

plt.subplot(2, 5, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 5, 2)
plt.title('Edges Detected')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 5, 3)
plt.title('Mask')
plt.imshow(mask, cmap='gray')

plt.subplot(2, 5, 4)
plt.title('Without Edges')
plt.imshow(image_without_edges, cmap='gray')

plt.subplot(2, 5, 5)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')

plt.subplot(2, 5, 6)
plt.title('Top Hat')
plt.imshow(top_hat, cmap='gray')

plt.subplot(2, 5, 7)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 5, 8)
plt.title('Opened Image')
plt.imshow(opened, cmap='gray')

cv2.imshow('edge_mask', cv2.resize(mask, None, fx=0.3, fy=0.3))
cv2.imwrite('../okokok.png', opened)
plt.tight_layout()
plt.show()
