import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

cnt = 0
error = []

def process_roi(img, points):
    x, y, w, h = cv2.boundingRect(points)
    roi_crop = img[y:y+h, x:x+w]
    # cv2.imshow('roi_crop', cv2.resize(roi_crop, None, fx=0.2, fy=0.2))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return roi_crop

def calculate_blurriness(img):
    # Check if the image has more than one channel (i.e., it's not grayscale)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var


def test():
    global p1, p2, p3, cnt
    import os
    from collections import defaultdict
    record = defaultdict(list)  # key = var // 10, value = a list of file's name
    high = 0
    low = sys.maxsize
    current_path = os.getcwd()
    files = os.listdir(current_path)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    for file in files:
        if file.lower().endswith(image_extensions):
            cnt += 1
            print(f"已經處理圖片數：{cnt}")
            print(f"當前圖片名稱：{file}")
            # image_processed_test(cv2.imread(file), p2)
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            var = calculate_blurriness(process_roi(img, p2))
            print(f'當前圖片 var = {var}')
            k = var // 100
            record[k].append(file)
            if var > high:
                high = var
            elif var < low:
                low = var
    if len(error) > 0:
        print('積屑量圖片如下：')
        print(error)

    print(f'high = {high}')
    print(f'low = {low}')
    for k in sorted(record):
        print(k, record[k])
    # for k, v in record.items():
    #     print(k, v)
    os.system('pause')
    # photo_20240718_154919


if __name__ == '__main__':
    # img = cv2.imread('photo_20240718_154919.png')
    # image_processed(img, p2)
    import os
    # os.chdir('C:\\Users\\natsumi\\Desktop\\專題相關\\模糊\\1_b\\新')
    print(os.getcwd())
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test()
