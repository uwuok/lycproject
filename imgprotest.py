import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 以灰階圖像讀入
    image = cv2.imread('C:\\Users\\natsumi\\PycharmProjects\\pythonProject\\image\\data sample\\1\\photo_20240628_105242_roi2.png', cv2.IMREAD_UNCHANGED)
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.bilateralFilter(image, 5, 50, 100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny 檢測邊緣
    edges = cv2.Canny(image, 350, 1000)  # 邊緣為白色(1)

    # 膨胀边缘，使其更厚
    kernel_dilate_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated_edges = cv2.dilate(edges, kernel_dilate_edges, iterations=11)

    # 二值化
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    binary_image = cv2.bitwise_not(binary_image)
    
    # 使用膨胀后的边缘作为掩码去除二值化图像中的外轮廓
    mask = cv2.bitwise_not(dilated_edges)
    binary_image_without_edges = cv2.bitwise_and(binary_image, binary_image, mask=mask)

    # 轮廓填充
    # contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(binary_image_without_edges, contours, -1, (0), thickness=cv2.FILLED)

    # 形态学腐蚀，获得整体切屑轮廓
    # ksize = 2, 2
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary_image_without_edges, kernel_erode, iterations=1)

    # 形态学膨胀，获得整个切屑特征
    # ksize = 3, 3
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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
    plt.title('Dilated Edges')
    plt.imshow(dilated_edges, cmap='gray')

    plt.subplot(2, 5, 4)
    plt.title('Binary Image')
    plt.imshow(binary_image, cmap='gray')

    plt.subplot(2, 5, 5)
    plt.title('Binary Image without Edges')
    plt.imshow(binary_image_without_edges, cmap='gray')

    plt.subplot(2, 5, 6)
    plt.title('Eroded Image')
    plt.imshow(eroded, cmap='gray')

    plt.subplot(2, 5, 7)
    plt.title('Dilated Image (result)')
    plt.imshow(dilated, cmap='gray')

    # 保存最终结果
    cv2.imwrite('final_image.png', dilated)

    plt.tight_layout()
    plt.show()
