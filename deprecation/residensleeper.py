import cv2
import numpy as np
import matplotlib.pyplot as plt


def c():
    points = np.array([[0, 0], [0, 11], [10, 10], [10, 0]])
    a = cv2.contourArea(points)
    print("area a is:", a)



def draw_poly(img, points):
    cv2.polylines(img, [np.array(points)], isClosed=True, color=(255, 255, 255), thickness=10)
    return img


def process_edge(img, points):
    mask = np.zeros_like(img)
    edge = draw_poly(mask, points)
    x, y, w, h = cv2.boundingRect(points)
    return edge[y:y + h, x:x + w]


def process_roi(img, points):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    return masked_img[y:y + h, x:x + w]


def image_processed(img, points):
    # 1. 轉換成灰階
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 提取 ROI 圖像以及邊緣
    roi = process_roi(img, points)
    edge = process_edge(img, points)

    print(calculate_blurriness(roi))

    # 直方圖等化提高對比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
    eq = clahe.apply(roi)

    # 雙邊濾波
    # bf = cv2.bilateralFilter(eq, 3, 25, 50)
    bf = cv2.GaussianBlur(eq, (5, 5), 0)
    # top hat
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(bf, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    # th = cv2.morphologyEx(bf, cv2.MORPH_BLACKHAT, kernel)

    # 二值化
    # _, b = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -1)
    # 去除邊緣（如果需要）
    # rm_edge = cv2.bitwise_xor(b, edge)

    # 形態學腐蝕，獲得整體切屑輪廓
    # good
    # eroded = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3)
    eroded = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    # dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    # eroded = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
    # 膨脹
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)), iterations=4)

    # 計算 ROI 面積
    roi_area = cv2.contourArea(points)
    white_pixels = cv2.countNonZero(dilated)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    # cv2.imshow('b', cv2.resize(b, None, fx=0.1, fy=0.1))

    # 計算區域內的非零值的個數
    # mask = np.ones_like(dilated)
    # print(cv2.countNonZero(mask))  # 14727801
    #
    # # 計算區域內的面積
    # height, width = mask.shape
    # ps = np.array([[0, 0], [0, width], [height, width], [height, 0]])
    # print(cv2.contourArea(ps))  # # 14727801.0


    print(f"ROI 面積: {roi_area} 像素")
    print(f"白色像素： {white_pixels}")
    print(f"黑色像素： {black_pixels}")
    print(f"白色像素比例: {white_ratio:.2%}")
    print(f"黑色像素比例: {black_ratio:.2%}")
    #
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()
    #
    # 顯示結果
    titles = ['Grayscale', 'ROI', 'Edge', 'Equalized', 'Top Hat',
              'Binary', 'eroded', 'dilated']
    images = [img, roi, edge, eq, th, b, eroded, dilated]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def calculate_blurriness(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()
    gray = cv2.resize(gray, None, fx=0.1, fy=0.1)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

if __name__ == '__main__':
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
    # 747 good
    # image = cv2.imread('new2.png')
    # 798 good
    # image = cv2.imread('photo_20240718_153843.png')
    # 3016 bad (過曝) top hat 結果過好，需改進直方圖等化 或 再加上模糊
    # image = cv2.imread('photo_20240718_153508.png')
    # 1344 bad (過曝) top hat 效果可，二值化後過曝
    # image = cv2.imread('photo_20240718_154919.png')
    # 629 bad (只找到部分積屑) top hat 效果差
    # image = cv2.imread('photo_20240718_155240.png')
    # image_processed(image, p2)


    # f(image)
    # c()
    import os
    print(os.getcwd())
    # C:\Users\natsumi\PycharmProjects\pythonProject\image
