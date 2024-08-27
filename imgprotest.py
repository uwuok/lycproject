import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawpoly(img, points):
    cv2.polylines(img, [np.array(points)], isClosed=True, color=(255, 255, 255), thickness=10)
    return img


def process_edge(img, points):
    img = np.zeros_like(img)
    mask = drawpoly(img, points)
    x, y, w, h = cv2.boundingRect(points)
    return mask[y:y+h, x:x+w]



def process_roi(img, points):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    return masked_img[y:y+h, x:x+w]


def image_processed(img):
    p1 = np.array([[357, 502], [1192, 552], [1112, 607],
                   [1012, 682], [932, 747], [852, 832],
                   [782, 927], [707, 1032], [657, 1127],
                   [622, 1187], [592, 1262], [557, 1342],
                   [532, 1412], [510, 1490], [490, 1595],
                   [475, 1720]])
    p2 = np.array([[10, 2530], [610, 2590], [690, 2810],
                   [790, 2990], [890, 3150], [970, 3290],
                   [1130, 3450], [1250, 3570], [1370, 3650],
                   [1430, 3710], [1550, 3790], [1610, 3850],
                   [1790, 3960], [2070, 5560], [190, 5680]])
    p3 = np.array([[2870, 3210], [6090, 2790], [7190, 4610],
                   [3010, 5990], [2630, 4310], [2590, 3970],
                   [2610, 3770], [2650, 3610], [2670, 3530],
                   [2730, 3410]])
    # 1.轉換成灰階
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2.進行雙邊模糊 (提升邊緣明顯度，由於已有邊的點，因此可以省略或改成高斯模糊等，方便後續處理)
    # img = cv2.bilateralFilter(img, 5, 50, 100)

    # 2. 提取 ROI 圖像 以及 邊緣
    a1 = process_roi(img, p1)
    e1 = process_edge(img, p1)
    a2 = process_roi(img, p2)
    e2 = process_edge(img, p2)
    a3 = process_roi(img, p3)
    e3 = process_edge(img, p3)

    cv2.imshow('a1', cv2.resize(a1, None, fx=0.4, fy=0.4))
    cv2.imshow('a2', cv2.resize(a2, None, fx=0.1, fy=0.1))
    cv2.imshow('a3', cv2.resize(a3, None, fx=0.1, fy=0.1))

    # 3. 直方圖等化提高對比度
    # 4. 自適應二值化
    # 5. 去除邊緣
    # 6. top hat
    # 7.

def image_processed_test(img, points):
    # 以灰階圖像讀入
    # image = cv2.imread('C:\\Users\\natsumi\\PycharmProjects\\pythonProject\\image\\data sample\\1\\photo_20240628_105242_roi2.png', cv2.IMREAD_UNCHANGED)
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.imread()
    image = cv2.bilateralFilter(image, 5, 50, 100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.imread('deprecation/edge_mask/new_edges_mask_1.png')

    # Canny 檢測邊緣
    # edges = cv2.Canny(image, 350, 1000)  # 邊緣為白色(1)

    # 膨胀边缘，使其更厚
    kernel_dilate_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated_edges = cv2.dilate(edges, kernel_dilate_edges, iterations=1)

    # 二值化
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    binary_image = cv2.bitwise_not(binary_image)

    # 使用膨胀后的边缘作为掩码去除二值化图像中的外轮廓
    mask = cv2.bitwise_not(dilated_edges)
    binary_image_without_edges = cv2.bitwise_and(binary_image, binary_image, mask=mask)

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


if __name__ == '__main__':
    image = cv2.imread('new2.png')
    image_processed(image)
    # f(image)
