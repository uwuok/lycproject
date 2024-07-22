import cv2
import numpy as np

# 讀取影像
image = cv2.imread('3.png')

# 創建一個與影像尺寸相同的遮罩，初始為全黑
mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
mask3 = np.zeros(image.shape[:2], dtype=np.uint8)

# 區塊一
points1 = np.array([[0, 2357], [45, 5959], [2115, 5774], [1655, 3684], [433, 2336]])
# points1 = np.array([[132, 65], [270, 1430], [595, 600], [1134, 200]])
# 區域二
points2 = np.array([[159, 102], [1130, 218], [306, 1469]])
# 區域三
points3 = np.array([[2988, 5944], [2453, 3779], [2727, 3008], [6032, 2584], [7113, 4496]])

# 繪製多邊形
cv2.fillPoly(mask1, [points1], 255)
cv2.fillPoly(mask2, [points2], 255)
cv2.fillPoly(mask3, [points3], 255)

# 應用遮罩，提取ROI
roi_1 = cv2.bitwise_and(image, image, mask=mask1)
roi_2 = cv2.bitwise_and(image, image, mask=mask2)
roi_3 = cv2.bitwise_and(image, image, mask=mask3)

# 創建一個與原圖相同大小的畫布
canvas = np.zeros_like(image)

# 將ROI按照原始位置放置在畫布上
canvas[mask1 > 0] = roi_1[mask1 > 0]
canvas[mask2 > 0] = roi_2[mask2 > 0]
canvas[mask3 > 0] = roi_3[mask3 > 0]

# 顯示拼接後的圖像
cv2.imshow('Combined ROI in Original Position', cv2.resize(canvas, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
