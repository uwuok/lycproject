import cv2
import numpy as np
import matplotlib.pyplot as plt

def contour(img, thresh):
    # 找出輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製輪廓
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.show()


def m():

    # 讀取灰階圖像
    img = cv2.imread('../image/dst2.jpg', 0)

    # 固定閾值化
    _, binary_fixed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu 的閾值化
    otsu_thresh, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 自適應閾值化  11, 2
    adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -11)
    adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    contour(img, adaptive_mean)
    contour(img, adaptive_gaussian)
    # 顯示結果
    scale_factor = 0.2
    cv2.imshow('Original', cv2.resize(img, None, fx=scale_factor, fy=scale_factor))
    # cv2.imshow('Fixed Thresholding', cv2.resize(binary_fixed, None, fx=scale_factor, fy=scale_factor))
    # cv2.imshow('Otsu Thresholding', cv2.resize(binary_otsu, None, fx=scale_factor, fy=scale_factor))
    cv2.imshow('Adaptive Mean Thresholding', cv2.resize(adaptive_mean, None, fx=scale_factor, fy=scale_factor))
    cv2.imshow('Adaptive Gaussian Thresholding', cv2.resize(adaptive_gaussian, None, fx=scale_factor, fy=scale_factor))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    m()

    # 讀取圖像並轉換為灰度圖像
    img = cv2.imread('dst2.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顯示灰度圖像
    plt.imshow(img_gray, cmap='gray')
    plt.title('Gray Image')
    plt.show()

    # 自適應閾值處理
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 顯示二值化後的圖像
    plt.imshow(thresh, cmap='gray')
    plt.title('Adaptive Threshold Image')
    plt.show()

    # 找出輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製輪廓
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    # 顯示輪廓
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.show()

    # 計算每個輪廓的面積
    areas = [cv2.contourArea(contour) for com4ntour in contours]
    total_area = sum(areas)

    print(f"Total Area of Chips: {total_area}")
