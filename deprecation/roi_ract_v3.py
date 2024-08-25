import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('3.png')

# x = 138, y = 115
# x = 1119, y = 198
# x = 1131, y = 1564
# x = 281, y = 1523
points1 = np.array([[138, 115], [1119, 198], [1131, 1564], [281, 1523]], dtype=np.float32)

# new_points1 = get_bounding_box_corners(points1)
# (0, 0), (1485, 0), (1485, 1050), (0, 1050)
# new_points1 = np.array([[0, 0], [1050, 0], [1050, 1485], [0, 1485]], dtype=np.float32)
new_points1 = np.array([[0, 0], [8000, 0], [8000, 6000], [0, 6000]], dtype=np.float32)

m1 = cv2.getPerspectiveTransform(points1, new_points1)
width1, height1 = np.max(new_points1[:, 0]), np.max(new_points1[:, 1])
dst1 = cv2.warpPerspective(image, m1, (int(width1), int(height1)))

cv2.imshow('input', cv2.resize(image, None, fx=0.1, fy=0.1))
cv2.imshow('area1', cv2.resize(dst1, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
plt.show()
