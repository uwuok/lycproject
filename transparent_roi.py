import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('deprecation/4.png', cv2.IMREAD_UNCHANGED)

# 定義多邊形的頂點
points = np.array([[100, 100], [200, 100], [250, 200], [150, 250]], np.int32)
points = points.reshape((-1, 1, 2))

# 創建掩碼
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [points], 255)

# 創建一個輸出圖像，將非多邊形區域設為透明
output = np.zeros_like(image, dtype=np.uint8)
output[mask == 255] = image[mask == 255]

# 如果圖像沒有 alpha 通道，則添加一個
if image.shape[2] == 3:
    output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)

# 將非多邊形區域的 alpha 通道設為0（透明）
output[mask == 0, 3] = 0

# 找到多邊形區域的邊界框
x, y, w, h = cv2.boundingRect(points)

# 裁切圖像以只保留多邊形區域
cropped_output = output[y:y+h, x:x+w]

# 保存結果
cv2.imwrite('output.png', cropped_output)
