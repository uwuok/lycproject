import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import json

ms1 = np.array([[7, 38], [69, 207], [146, 376], [176, 438], [253, 546],
                [330, 653], [376, 730], [438, 807], [484, 869], [546, 930],
                [607, 992], [684, 1053], [761, 1130], [823, 1176], [884, 1223],
                [992, 1315], [1053, 1346], [1130, 1392], [1192, 1423],
                [1238, 1453], [1238, 1576], [1269, 1807], [1300, 2084],
                [1315, 2207], [1346, 2376], [1469, 3238], [346, 3469]])

ms2 = np.array([[0, 0], [538, 0], [894, 74], [798, 134],
                [766, 158], [698, 202], [622, 266], [578, 310],
                [522, 362], [426, 478], [374, 550], [318, 626],
                [294, 674], [254, 758], [210, 842], [182, 910],
                [158, 982], [134, 1078], [130, 1114], [22, 1114],
                [0, 974]])

def update_json(value):
    file_path = r'config.json'
    
    # 檢查文件是否存在，如果不存在則創建一個空的 JSON 文件
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({}, file)
    
    try:
        # 讀取現有的 JSON 數據
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # 更新正確的 key
        data['Flusher_level_bar'] = value # 積屑占比
        level = 0
        if value < 0.21:
            level = 1
        elif 0.21 <= value < 0.41:
            level = 2
        elif 0.41 <= value < 0.61:
            level = 3
        elif 0.61 <= value < 0.81:
            level = 4
        elif 0.81 <= value <= 1:
            level = 5
        data['Flusher_level'] = level
        
        # 將更新的內容寫入到 JSON 文件
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_white_ratio(img, ps):
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    roi_area = np.count_nonzero(mask)
    roi = cv2.bitwise_and(img, img, mask=mask)
    if len(roi.shape) > 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white_pixels = cv2.countNonZero(roi)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area
    # print(f"ROI 面積: {roi_area} 像素")
    # print(f"ROI 白色像素： {white_pixels}")
    # print(f"ROI 黑色像素： {black_pixels}")
    print(f"ROI 白色像素比例: {white_ratio:.2%}")
    # print(f"ROI 黑色像素比例: {black_ratio:.2%}")
    update_json(white_ratio)
    return white_ratio

def mask_roi(img, ps):
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    # cv2.imshow('mask', cv2.resize(mask, None, fx=0.2, fy=0.2))
    # cv2.waitKey()
    roi_area = np.count_nonzero(mask)
    roi = cv2.bitwise_and(img, img, mask=mask)
    if len(roi.shape) > 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return roi


def get_roi(image):
    # image = cv2.imread(r"C:\Users\natsumi\PycharmProjects\pythonProject\image\photo_20240718_153508.png")
    # x, y
    pts1 = np.array([[560, 2539], [1550, 2504], [2053, 5699], [880, 5982]], np.int32)
    pts2 = np.array([[370, 410], [1341, 466], [1323, 1515], [488, 1528]], np.int32)

    # 最小邊界矩形並擷取 ROI
    x1, y1, w1, h1 = cv2.boundingRect(pts1)
    roi1 = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = cv2.boundingRect(pts2)
    roi2 = image[y2:y2 + h2, x2:x2 + w2]
    return roi1, roi2


def proc_0919(img, timestamp):
    current_img = img
    roi1, roi2 = get_roi(current_img)  # 獲取兩個 ROI 區域
    # 灰階處理
    g1, g2 = roi1, roi2
    if len(roi1.shape) == 3:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    if len(roi2.shape) == 3:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    m1 = cv2.medianBlur(g1, 7)
    m2 = cv2.medianBlur(g2, 7)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    cv2.imshow('m1', m1)
    cv2.imshow('m2', m2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    c1 = clahe.apply(m1)
    c2 = clahe.apply(m2)
    
    t1 = cv2.adaptiveThreshold(
        c1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -4)
    t2 = cv2.adaptiveThreshold(
        c2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -4)
    d1 = cv2.dilate(t1, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=2)
    d2 = cv2.dilate(t2, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)), iterations=2)
    r1 = mask_roi(d1, ms1)
    r2 = mask_roi(d2, ms2)
    f1 = f"photo_{timestamp}_1.png"
    f2 = f"photo_{timestamp}_2.png"
    calculate_white_ratio(r1, ms1)
    # cv2.imwrite(f1, r2)
    # cv2.imwrite(f2, r2)
    # print(f"已保存 ROI1 為： {f1}")
    # print(f"已保存 ROI2 為： {f2}")
    return r1, r2


if __name__ == '__main__':
    print('123')
