import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
# points = []


def mask_roi(roi1, roi2):
    ps1 = np.array([[7, 38],
                    [69, 207],
                    [146, 376],
                    [176, 438],
                    [253, 546],
                    [330, 653],
                    [376, 730],
                    [438, 807],
                    [484, 869],
                    [546, 930],
                    [607, 992],
                    [684, 1053],
                    [761, 1130],
                    [823, 1176],
                    [884, 1223],
                    [992, 1315],
                    [1053, 1346],
                    [1130, 1392],
                    [1192, 1423],
                    [1238, 1453],
                    [1238, 1576],
                    [1269, 1807],
                    [1300, 2084],
                    [1315, 2207],
                    [1346, 2376],
                    [1469, 3238],
                    [346, 3469]])
    ps2 = np.array([[0, 0],
                    [950, 60],
                    [882, 104],
                    [818, 144],
                    [734, 196],
                    [674, 244],
                    [598, 308],
                    [530, 380],
                    [458, 460],
                    [410, 520],
                    [358, 592],
                    [326, 644],
                    [282, 716],
                    [250, 780],
                    [230, 828],
                    [202, 892],
                    [182, 936],
                    [162, 1000],
                    [154, 1040],
                    [138, 1096],
                    [134, 1116],
                    [78, 1116]])
    mask_shape = roi1.shape[:2]
    m1 = np.zeros(mask_shape, dtype=np.uint8)
    mask_shape = roi2.shape[:2]
    m2 = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(m1, [ps1], 255)
    cv2.fillPoly(m2, [ps2], 255)
    roi1 = cv2.bitwise_and(roi1, roi1, mask=m1)
    roi2 = cv2.bitwise_and(roi2, roi2, mask=m2)
    roi_area = cv2.contourArea(ps1)
    white_pixels = cv2.countNonZero(roi1)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    print(f"ROI1 面積: {roi_area} 像素")
    print(f"ROI1 白色像素： {white_pixels}")
    print(f"ROI1 黑色像素： {black_pixels}")
    print(f"ROI1 白色像素比例: {white_ratio:.2%}")
    print(f"ROI1 黑色像素比例: {black_ratio:.2%}")

    roi_area = cv2.contourArea(ps2)
    white_pixels = cv2.countNonZero(roi2)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    print(f"ROI2 面積: {roi_area} 像素")
    print(f"ROI2 白色像素： {white_pixels}")
    print(f"ROI2 黑色像素： {black_pixels}")
    print(f"ROI2 白色像素比例: {white_ratio:.2%}")
    print(f"ROI2 黑色像素比例: {black_ratio:.2%}")

    return roi1, roi2


def get_roi(image):
    pts1 = np.array([[560, 2539], [1550, 2504], [
        2053, 5699], [880, 5982]], np.int32)

    pts2 = np.array([[370, 410], [1341, 466], [
                    1323, 1515], [488, 1528]], np.int32)

    # 最小邊界矩形並擷取 ROI
    x1, y1, w1, h1 = cv2.boundingRect(pts1)
    roi1 = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = cv2.boundingRect(pts2)
    roi2 = image[y2:y2 + h2, x2:x2 + w2]
    print(f'pts1 = [[0, 0], [0, {w1}], [{w1}, {h1}], [{h1}, 0]]')
    print(f'pts1 = [[0, 0], [0, {w2}], [{w2}, {h2}], [{h2}, 0]]')
    return roi1, roi2


def image_processed(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_median = cv2.medianBlur(gray_img, 7)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_median)
    filterSize = (8, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    threshold = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    re = cv2.dilate(threshold, kernel, iterations=2)
    closing = cv2.morphologyEx(re, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing


if __name__ == '__main__':
    image = cv2.imread(r"C:\Users\User\Desktop\1\photo_20240903_162345.png")
    roi1, roi2 = get_roi(image)
    cv2.imshow('roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    proc_roi1 = image_processed(roi1)
    proc_roi2 = image_processed(roi2)
    res1, res2 = mask_roi(proc_roi1, proc_roi2)
    cv2.imshow('1', cv2.resize(res1, None, fx=0.2, fy=0.2))
    cv2.imshow('2', cv2.resize(res2, None, fx=0.2, fy=0.2))
    cv2.imwrite(r"C:\Users\User\Desktop\1\roi1\roi1.png", res1)
    cv2.imwrite(r"C:\Users\User\Desktop\1\roi1\roi2.png", res2)
    cv2.waitKey()
    cv2.destroyAllWindows()
