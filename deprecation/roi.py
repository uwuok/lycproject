import cv2
import numpy as np

# 讀取影像
image = cv2.imread('4.png')

# 創建一個與影像尺寸相同的遮罩，初始為全黑
mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
mask3 = np.zeros(image.shape[:2], dtype=np.uint8)

# 區塊一
# 0, 2357 (左上)
# 433, 2336  (右上)
# 45, 5959   (左下)
# 2115, 5774 (右下)
# 1655, 3684 (右中)

# 區域二
# 159, 102 (左上)
# 1130, 218 (右上)
# 306, 1469 (左下)

# 區域三
# 2988, 5944　(左下)
# 2453, 3770  (左中)
# 2727, 3008  (左上)
# 6032, 2584  (右上)
# 7113, 4496  (右下)


# 指定五個頂點
# points1 = np.array([[2357, 0], [5959, 45], [5774, 2115], [3684, 1655], [2336, 433]])
points1 = np.array([[0, 2357], [45, 5959], [2115, 5774], [1655, 3684], [433, 2336]])
points2 = np.array([[159, 102], [1130, 218], [306, 1469]])
points3 = np.array([[2988, 5944], [2453, 3779], [2727, 3008], [6032, 2584], [7113, 4496]])

# 繪製多邊形
cv2.fillPoly(mask1, [points1], 255)
cv2.fillPoly(mask2, [points2], 255)
cv2.fillPoly(mask3, [points3], 255)

# 應用遮罩，提取ROI
roi_1 = cv2.bitwise_and(image, image, mask=mask1)
roi_2 = cv2.bitwise_and(image, image, mask=mask2)
roi_3 = cv2.bitwise_and(image, image, mask=mask3)

# 顯示原影像、遮罩和ROI
cv2.imshow('Original Image', cv2.resize(image, None, fx=0.1, fy=0.1))
# cv2.imshow('Mask', cv2.resize(edge_mask, None, fx=0.1, fy=0.1))
cv2.imshow('roi_1', cv2.resize(roi_1, None, fx=0.1, fy=0.1))
cv2.imshow('roi_2', cv2.resize(roi_2, None, fx=0.1, fy=0.1))
cv2.imshow('roi_3', cv2.resize(roi_3, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
