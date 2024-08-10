import cv2
import numpy as np
import os

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
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return []  # 如果图像加载失败，返回空列表
    roi1 = get_roi_pic(img, points_1)
    roi2 = get_roi_pic(img, points_2)
    roi3 = get_roi_pic(img, points_3)

    rois = [roi1, roi2, roi3]
    roi_paths = []
    for i, roi in enumerate(rois):
        output_path = os.path.splitext(image_path)[0] + f'_roi{i + 1}.png'
        cv2.imwrite(output_path, roi)
        roi_paths.append(output_path)
        print(f"ROI image saved to {output_path}")
    return roi_paths

def process_image(image_path):
    # 以灰階圖像讀入
    image = cv2.imread('C:\\Users\\natsumi\\PycharmProjects\\pythonProject\\image\\data sample\\1\\photo_20240628_105242_roi2.png', cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
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

    # 將結果儲存至原始圖片所在的資料夾中
    output_path = os.path.splitext(image_path)[0] + '_processed.png'
    cv2.imwrite(output_path, dilated)
    print(f"Processed image saved to {output_path}")

def process_images_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                roi_paths = roi(image_path)
                for roi_path in roi_paths:
                    # 檢查檔案是否以 'roi' 作為後綴，並對其進行處理
                    if any(roi_path.endswith(f'_roi{i}.png') for i in range(1, len(roi_paths) + 1)):
                        process_image(roi_path)

if __name__ == '__main__':
    # 設定要處理的資料夾路徑
    folder_path = '.'  # 當前資料夾
    process_images_in_folder(folder_path)
