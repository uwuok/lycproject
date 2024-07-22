import cv2
import numpy as np

# 讀取灰階圖像
img = cv2.imread('../image/10.png', 0)

# 固定閾值化
_, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu 的閾值化
_, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 自適應閾值化
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 顯示結果
scale_factor = 0.1
cv2.imshow('Original', cv2.resize(img, None, fx=scale_factor, fy=scale_factor))
cv2.imshow('Fixed Thresholding', cv2.resize(binary_fixed, None, fx=scale_factor, fy=scale_factor))
cv2.imshow('Otsu Thresholding', cv2.resize(binary_otsu, None, fx=scale_factor, fy=scale_factor))
cv2.imshow('Adaptive Mean Thresholding', cv2.resize(adaptive_mean, None, fx=scale_factor, fy=scale_factor))
cv2.imshow('Adaptive Gaussian Thresholding', cv2.resize(adaptive_gaussian, None, fx=scale_factor, fy=scale_factor))
cv2.waitKey(0)
cv2.destroyAllWindows()
