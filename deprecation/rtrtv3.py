import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


import cv2
import numpy as np

points_1 = np.array([[148, 200], [1030, 248], [766, 441],
                     [627, 589], [490, 780], [404, 949],
                     [325, 1167], [269, 1411]])

points_2 = np.array([[380, 2447], [456, 2445], [593, 2709],
                     [904, 3156], [1106, 3325], [1301, 3489],
                     [1639, 3688], [2017, 5446], [799, 5788]])

points_3 = np.array([[2737, 2997], [5957, 2595], [6948, 4363],
                     [2928, 5754], [2487, 4061], [2454, 3595],
                     [2528, 3261], [2626, 3097]])

def get_roi_pic(image, points):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask_shape = image.shape[:2]  # 原圖大小
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Extract ROI
    roi_img = cv2.bitwise_and(image, image, mask=mask)

    # Find bounding box of the ROI
    x, y, w, h = cv2.boundingRect(points)

    # Create black background with size of bounding box
    black_bg = np.zeros((h, w, image.shape[2]), dtype=np.uint8)

    # Copy ROI region to the black background
    roi_crop = roi_img[y:y+h, x:x+w]
    black_bg[:h, :w] = roi_crop

    return black_bg


def roi(image_path):
    # img = cv2.imread('3.png')
    img = cv2.imread(image_path)
    # mask_shape = img.shape[:2]
    roi1 = get_roi_pic(img, points_1)
    roi2 = get_roi_pic(img, points_2)
    roi3 = get_roi_pic(img, points_3)
    print(img.shape)
    dst1 = cv2.resize(roi1, dsize=(len(roi1[0]), len(roi1[2])), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('area1', cv2.resize(roi1, None, fx=0.5, fy=0.5))
    dst2 = cv2.resize(roi2, dsize=(roi2.shape[1], roi2.shape[0]), interpolation=cv2.INTER_AREA)
    # cv2.imshow('area2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    dst3 = cv2.resize(roi3, dsize=roi3.shape[1::-1], interpolation=cv2.INTER_AREA)
    # cv2.imshow('area3', cv2.resize(roi3, None, fx=0.2, fy=0.2))
    # contour(dst2, )
    # cv2.imwrite('dst1.jpg', dst1)
    # cv2.imwrite('dst2.jpg', dst2)
    # cv2.imwrite('dst3.jpg', dst3)
    # 將結果儲存至原始圖片所在的資料夾中
    pics = [dst1, dst2, dst3]
    for i in range(len(pics)):
        output_path = os.path.splitext(image_path)[0] + f'_dst{i + 1}.png'
        cv2.imwrite(output_path, pics[i])
    print(f"Processed image saved to {output_path}")
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def process_image(image_path):
    # 以灰階圖像讀入
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Canny 檢測邊緣
    edges = cv2.Canny(image, 350, 900)

    # 先找到原圖的邊緣 (以便移除後續二值化所產生的邊緣)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 自適應直方圖均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)

    # top hat 將暗處的亮點提升
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(equalized_image, cv2.MORPH_TOPHAT, kernel)

    # 二值化
    _, binary_image = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.drawContours(binary_image, contours, -1, (0), thickness=2)

    # 形態學腐蝕，獲得整體切屑輪廓
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary_image, kernel_erode, iterations=1)

    # 形態學膨脹，獲得整個切屑特徵
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)

    # 將結果儲存至原始圖片所在的資料夾中
    output_path = os.path.splitext(image_path)[0] + '_processed.png'
    cv2.imwrite(output_path, dilated)
    print(f"Processed image saved to {output_path}")

def process_images_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                roi(image_path)
                # process_image(image_path)


if __name__ == '__main__':
    # 設定要處理的資料夾路徑
    folder_path = '..'  # 當前資料夾
    process_images_in_folder(folder_path)
