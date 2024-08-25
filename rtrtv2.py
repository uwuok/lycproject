import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 以灰階圖像讀入
    # image = cv2.imread('area2_2.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

    # Canny 檢測邊緣
    edges = cv2.Canny(image, 350, 900)

    # 先找到原圖的邊緣 (以便移除後續二值化所產生的邊緣)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 直方圖均衡化
    # equalized_image = cv2.equalizeHist(image_without_edges)
    # 自適應直方圖均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)

    # top hat 將暗處的亮點提升
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)

    # 二值化
    _, binary_image = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.drawContours(binary_image, contours, -1, (0), thickness=2)

    # 形态学腐蚀，获得整体切屑轮廓
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary_image, kernel_erode, iterations=1)

    # 形态学膨胀，获得整个切屑特征
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)

    # 显示结果
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 5, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 5, 2)
    plt.title('Edges Detected')
    plt.imshow(edges, cmap='gray')

    plt.subplot(2, 5, 3)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')

    plt.subplot(2, 5, 4)
    plt.title('Top Hat')
    plt.imshow(top_hat, cmap='gray')

    plt.subplot(2, 5, 5)
    plt.title('Binary Image')
    plt.imshow(binary_image, cmap='gray')

    plt.subplot(2, 5, 6)
    plt.title('Eroded Image')
    plt.imshow(eroded, cmap='gray')

    plt.subplot(2, 5, 7)
    plt.title('Dilate Image (result)')
    plt.imshow(dilated, cmap='gray')

    # cv2.imshow('edge_mask', cv2.resize(edge_mask, None, fx=0.3, fy=0.3))
    # cv2.imwrite('okokok.png', dilated)
    plt.tight_layout()
    plt.show()
    # cv2.imwrite('rtrtv2.png', dilated)

