import cv2
import numpy as np
import matplotlib.pyplot as plt


def c():
    points = np.array([[0, 0], [0, 11], [10, 10], [10, 0]])
    a = cv2.contourArea(points)
    print("area a is:", a)



def process_edge(img, points):
    mask = np.zeros_like(img)
    # edge = draw_poly(mask, points)
    if len(img.shape) == 3:
        edge = cv2.polylines(mask, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=1)
    else:
        edge = cv2.polylines(mask, [np.array(points)], isClosed=True, color=255, thickness=1)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    x, y, w, h = cv2.boundingRect(points)
    return edge[y:y + h, x:x + w]


def process_roi(img, points):
    mask = np.zeros_like(img)
    if len(img.shape) == 3:
        cv2.fillPoly(mask, [points], (255, 255, 255))
    else:
        cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(points)
    return masked_img[y:y + h, x:x + w]


def image_processed(img, points):
    # 1. 轉換成灰階
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 提取 ROI 圖像以及邊緣
    roi = process_roi(img, points)
    edge = process_edge(img, points)

    # 直方圖等化提高對比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    eq = clahe.apply(roi)

    # 雙邊濾波 降躁
    # bf = cv2.bilateralFilter(eq, 3, 25, 50)
    bf = cv2.GaussianBlur(eq, (21, 21), 0)
    # top hat
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(bf, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    # th = cv2.morphologyEx(bf, cv2.MORPH_BLACKHAT, kernel)

    # 二值化
    # _, b = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -1)
    # 使用 edge 去除邊緣（如果需要）
    # rm_edge = cv2.bitwise_xor(b, edge)
    # edge 黑底白邊 -> 白底黑邊
    edge = cv2.bitwise_not(edge)
    rm_edge = cv2.bitwise_and(b, edge)
    # 形態學腐蝕，獲得整體切屑輪廓
    # good
    # eroded = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3)
    eroded = cv2.erode(rm_edge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
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
    titles = ['Grayscale', 'ROI', 'Edge', 'Equalized', 'GaussianBlur', 'Top Hat',
              'Binary', 'rm edge', 'eroded', 'dilated']
    images = [img, roi, edge, eq, bf, th, b, rm_edge, eroded, dilated]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def contours_test(img, points):
    color_roi = process_roi(img, points)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = process_roi(gray, points)
    edge = process_edge(gray, points)
    bf = cv2.GaussianBlur(roi, (21, 21), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    # clahe = cv2.createCLAHE()
    eq = clahe.apply(bf)
    # eq = cv2.equalizeHist(roi)
    th = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    # 二值化
    b = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -1)
    # ret, b = cv2.threshold(th, 10, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    with_contours = cv2.drawContours(color_roi, contours, -1, (0, 255, 0), 10)
    edge = cv2.bitwise_not(edge)
    rm_edge = cv2.bitwise_and(b, edge)
    eroded = cv2.erode(rm_edge, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)), iterations=4)

    # 計算 ROI 面積
    roi_area = cv2.contourArea(points)
    white_pixels = cv2.countNonZero(dilated)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    print(f"ROI 面積: {roi_area} 像素")
    print(f"白色像素： {white_pixels}")
    print(f"黑色像素： {black_pixels}")
    print(f"白色像素比例: {white_ratio:.2%}")
    print(f"黑色像素比例: {black_ratio:.2%}")

    titles = ['ROI', 'Edge', 'Equalized', 'GaussianBlur', 'Top Hat',
              'Binary', 'with_contours', 'rm edge', 'eroded', 'dilated']
    images = [roi, edge, eq, bf, th, b, with_contours, rm_edge, eroded, dilated]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

    # cv2.imshow('gray img', cv2.resize(img, None, fx=0.1, fy=0.1))
    # cv2.imshow('binary', cv2.resize(b, None, fx=0.1, fy=0.1))
    # cv2.imshow('contours', cv2.resize(with_contours, None, fx=0.1, fy=0.1))
    cv2.waitKey()
    cv2.destroyAllWindows()


def blob_test(img, points):
    img = process_roi(img, points)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 設置參數
    params = cv2.SimpleBlobDetector_Params()
    # gray = cv2.GaussianBlur(gray, (101, 101), 0)
    # 篩選亮斑點
    # params.filterByColor = True
    # params.blobColor = 127  # 假設我們要檢測亮色的斑點

    # 設置篩選面積
    # params.filterByArea = True
    # params.minArea = 300  # 斑點的最小面積
    # params.maxArea = 3000  # 斑點的最大面積

    # 設置篩選圓度
    # params.filterByCircularity = True
    # params.minCircularity = 0.3  # 斑點的最小圓度

    # 創建檢測器
    detector = cv2.SimpleBlobDetector_create(params)

    # 檢測斑點
    keypoints = detector.detect(gray)

    # 在影像上繪製斑點
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # titles = ['res']
    # images = [im_with_keypoints]
    #
    # plt.figure(figsize=(15, 10))
    # for i in range(len(images)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.axis('off')
    # plt.show()

    # 顯示影像
    cv2.imshow('Blobs', cv2.resize(im_with_keypoints, None, fx=0.2, fy=0.2))
    cv2.imshow('res', cv2.resize(gray, None, fx=0.2, fy=0.2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    # image = cv2.imread('new2.png')
    # 1 清晰
    # image = cv2.imread('photo_20240718_153843.png')
    image = cv2.imread('photo_20240718_153508.png')
    # 1 模糊
    # image = cv2.imread('photo_20240718_154919.png')
    # 2 模糊
    # image = cv2.imread('photo_20240718_155240.png')
    # image_processed(image, p2)
    # image_processed(image, p2)
    # image_processed(image, p3)
    # f(image)
    # c()
    contours_test(image, p2)
    # blob_test(image, p2)
