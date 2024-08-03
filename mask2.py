import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 導入原始圖像
image = cv2.imread('area2_2.png')

# 2. 灰階轉換
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 直方圖等化
equalized_image = cv2.equalizeHist(gray_image)
# 3. 自適應直方圖等化（CLAHE）
# clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(20, 20))
# clahe_image = clahe.apply(gray_image)

# 4. 形態學 (Top Hat)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# tophat_image = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)
tophat_image = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)
tophat_image = cv2.GaussianBlur(tophat_image, (11, 11), 0)
# 5. 二值化
_, binary_image = cv2.threshold(tophat_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, binary_image = cv2.threshold(tophat_image, 77, 255, cv2.THRESH_BINARY)

# binary_image = cv2.adaptiveThreshold(tophat_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
# binary_image = cv2.medianBlur(binary_image, 15)
# print(_)

# 6. 形態學 (侵蝕) - 獲得整個切屑輪廓
dilated_image1 = cv2.erode(binary_image, kernel, iterations=1)

# 7. 形態學 (膨脹) - 獲得整個切屑特徵
dilated_image2 = cv2.dilate(dilated_image1, kernel, iterations=1)
# dilated_image2 = cv2.bitwise_not(dilated_image1)

# 8. 輸出圖片
plt.figure(figsize=(12, 6))

plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(232), plt.imshow(gray_image, cmap='gray'), plt.title('Gray Image')
plt.subplot(233), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
plt.subplot(245), plt.imshow(tophat_image, cmap='gray'), plt.title('Top Hat Image')
plt.subplot(246), plt.imshow(binary_image, cmap='gray'), plt.title('Binary Image')
plt.subplot(247), plt.imshow(dilated_image1, cmap='gray'), plt.title('Dilated Image')
plt.subplot(248), plt.imshow(dilated_image2, cmap='gray'), plt.title('Final Dilated Image')
cv2.imwrite('final_dilated_image.png', dilated_image2)
plt.tight_layout()
plt.show()
