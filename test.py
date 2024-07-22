import cv2
import numpy as np
from skimage.transform import ProjectiveTransform, warp

# 讀取影像
image = cv2.imread('3.png')

# 創建一個與影像尺寸相同的遮罩，初始為全黑
mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
mask3 = np.zeros(image.shape[:2], dtype=np.uint8)

# 區塊一
points1 = np.array([[0, 2357], [45, 5959], [2115, 5774], [1655, 3684], [433, 2336]])
# 區域二
points2 = np.array([[159, 102], [1130, 218], [306, 1469]])
# 區域三
points3 = np.array([[2988, 5944], [2453, 3779], [2727, 3008], [6032, 2584], [7113, 4496]])

# 繪製多邊形
cv2.fillPoly(mask1, [points1], 255)
cv2.fillPoly(mask2, [points2], 255)
cv2.fillPoly(mask3, [points3], 255)

# 提取ROI
roi_1 = cv2.bitwise_and(image, image, mask=mask1)
roi_2 = cv2.bitwise_and(image, image, mask=mask2)
roi_3 = cv2.bitwise_and(image, image, mask=mask3)

def fill_bounding_box(points, roi):
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Generate dst based on the number of points
    dst = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

    transform = ProjectiveTransform()
    if transform.estimate(points[:4], dst):
        output_shape = (max_y - min_y + 1, max_x - min_x + 1)
        warped = warp(roi, transform.inverse, output_shape=output_shape)
        return (min_x, min_y), warped
    else:
        raise Exception("Transformation estimation failed.")

# 將每個ROI填充到各自的最小外接矩形
(min_x1, min_y1), warped_roi_1 = fill_bounding_box(points1, roi_1)
# (min_x2, min_y2), warped_roi_2 = fill_bounding_box(points2, roi_2)
# (min_x3, min_y3), warped_roi_3 = fill_bounding_box(points3, roi_3)

# 創建一個與原圖相同大小的畫布
canvas = np.zeros_like(image)

# 將填充後的ROI放置到畫布上
canvas[min_y1:min_y1+warped_roi_1.shape[0], min_x1:min_x1+warped_roi_1.shape[1]] = (warped_roi_1 * 255).astype(np.uint8)
# canvas[min_y2:min_y2+warped_roi_2.shape[0], min_x2:min_x2+warped_roi_2.shape[1]] = (warped_roi_2 * 255).astype(np.uint8)
# canvas[min_y3:min_y3+warped_roi_3.shape[0], min_x3:min_x3+warped_roi_3.shape[1]] = (warped_roi_3 * 255).astype(np.uint8)

# 顯示拼接後的圖像
cv2.imshow('Combined ROI in Bounding Boxes', cv2.resize(canvas, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
