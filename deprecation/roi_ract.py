import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_min_bounding_box(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    width = max_x - min_x
    height = max_y - min_y
    return width, height


'''
Points1 -> Width: 1002, Height: 1365
Points2 -> Width: 2160, Height: 3500
Points3 -> Width: 4877, Height: 3480
'''


def get_bounding_box_corners(points):
    width, height = get_min_bounding_box(points)
    # y, x
    upper_left = [0, 0]
    upper_right = [0, width]
    lower_left = [height, 0]
    lower_right = [height, width]
    return upper_left, lower_left, lower_right, upper_right


# 讀取影像
image = cv2.imread('../3.png')

# area 1: 左上角
# 132, 65 左上
# 270, 1430 左下
# 595, 600 右下
# 1134, 200 右上

# area 2: 左下角
# 0, 2500 左上
# 0, 6000 左下
# 2160, 6000 右下
# 1434, 2727 右上

# area 3: 右下角
# 2268, 3031 左上
# 2955, 6000 左下
# 7145, 4525 右下
# 6243, 2520 右上

points1 = np.array([[132, 65], [270, 1430], [595, 600], [1134, 200]])
points2 = np.array([[0, 2500], [0, 6000], [2160, 6000], [1434, 2727]])
points3 = np.array([[2955, 6000], [7145, 4525], [6243, 2520], [2268, 3031]])

# 變換後的左上、右上、左下、右下的點
new_points1 = np.array(get_bounding_box_corners(points1))
new_points2 = np.array(get_bounding_box_corners(points2))
new_points3 = np.array(get_bounding_box_corners(points3))
# print(new_points1)
# print(new_points2)
# print(new_points3)

# 生成 perspective transformation matrix
m1 = cv2.getPerspectiveTransform(points1, new_points1)
m2 = cv2.getPerspectiveTransform(points2, new_points2)
m3 = cv2.getPerspectiveTransform(points3, new_points3)
# 進行 perspective transformation
dst = cv2.warpPerspective(image, m1, get_min_bounding_box(points1))

# 以 rgb 輸出
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')

# 顯示拼接後的圖像
# cv2.imshow('Combined ROI in Original Position', cv2.resize(canvas, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
