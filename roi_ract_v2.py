import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_min_bounding_box(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    w = max_x - min_x
    h = max_y - min_y
    return w, h


def get_bounding_box_corners(points):
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    upper_left = [min_x, min_y]
    upper_right = [max_x, min_y]
    lower_left = [min_x, max_y]
    lower_right = [max_x, max_y]
    return np.array([upper_left, lower_left, lower_right, upper_right], dtype=np.float32)

image = cv2.imread('3.png')

points1 = np.array([[132, 65], [270, 1430], [595, 600], [1134, 200]], dtype=np.float32)
points2 = np.array([[0, 2500], [0, 6000], [2160, 6000], [1434, 2727]], dtype=np.float32)
points3 = np.array([[2955, 6000], [7145, 4525], [6243, 2520], [2268, 3031]], dtype=np.float32)

# new_points1 = get_bounding_box_corners(points1)
# (0, 0), (1485, 0), (1485, 1050), (0, 1050)
new_points1 = np.array([[0, 0], [0, 1050], [1485, 1050], [1485, 0]], dtype=np.float32)
new_points2 = get_bounding_box_corners(points2)
new_points3 = get_bounding_box_corners(points3)

m1 = cv2.getPerspectiveTransform(points1, new_points1)
# width1, height1 = get_min_bounding_box(points1)
width1, height1 = np.max(new_points1[:, 0]), np.max(new_points1[:, 1])
dst1 = cv2.warpPerspective(image, m1, (int(width1), int(height1)))

m2 = cv2.getPerspectiveTransform(points2, new_points2)
width2, height2 = get_min_bounding_box(points2)
dst2 = cv2.warpPerspective(image, m2, (int(width2), int(height2)))
#
m3 = cv2.getPerspectiveTransform(points3, new_points3)
width3, height3 = get_min_bounding_box(points3)
dst3 = cv2.warpPerspective(image, m3, (int(width3), int(height3)))

cv2.imshow('input', cv2.resize(image, None, fx=0.1, fy=0.1))
cv2.imshow('area1', cv2.resize(dst1, None, fx=0.1, fy=0.1))
cv2.imshow('area2', cv2.resize(dst2, None, fx=0.1, fy=0.1))
cv2.imshow('area3', cv2.resize(dst3, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
plt.show()
