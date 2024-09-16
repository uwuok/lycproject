import cv2
import numpy as np

image = cv2.imread('photo_20240718_153843.png')
pts1 = np.array([[560, 2539], [1550, 2504], [
    2053, 5699], [880, 5982]], np.int32)

pts2 = np.array([[370, 410], [1341, 466], [1323, 1515], [488, 1528]], np.int32)

# 最小邊界矩形並擷取 ROI
x1, y1, w1, h1 = cv2.boundingRect(pts1)
roi1 = image[y1:y1+h1, x1:x1+w1]
x2, y2, w2, h2 = cv2.boundingRect(pts2)
roi2 = image[y2:y2+h2, x2:x2+w2]
print(f'pts1 = [[0, 0], [0, {w1}], [{w1}, {h1}], [{h1}, 0]]')
print(f'pts1 = [[0, 0], [0, {w2}], [{w2}, {h2}], [{h2}, 0]]')

# 轉換為灰度圖
gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)


def mean_brightness():
    # 計算平均亮度
    mean_brightness1 = np.mean(gray1)
    mean_brightness2 = np.mean(gray2)
    print("平均亮度1:", mean_brightness1)
    print("平均亮度2:", mean_brightness2)
# 平均亮度1: 131.2156905864331
# 平均亮度2: 149.56348168742667


# def hist():
#     # 計算灰度直方圖
#     hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
#     hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
#     # 計算直方圖重心
#     total_pixels1 = np.sum(hist1)
#     weighted_sum1 = sum(i * hist1[i] for i in range(256))
#     brightness_center1 = weighted_sum1 / total_pixels1
#
#     total_pixels2 = np.sum(hist2)
#     weighted_sum2 = sum(i * hist2[i] for i in range(256))
#     brightness_center2 = weighted_sum2 / total_pixels2
#
#     print(f"直方圖重心1: {brightness_center1[0]}")
#     print(f"直方圖重心2: {brightness_center2[0]}")

mean_brightness()
hist()
