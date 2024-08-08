import cv2
import numpy as np


points_2 = np.array([[380, 2447], [456, 2445], [593, 2709],
                     [904, 3156], [1106, 3325], [1301, 3489],
                     [1639, 3688], [2017, 5446], [799, 5788]])

# 讀取圖像
image = cv2.imread('../dst2.jpg')
height, width, _ = image.shape

# 定義不規則多邊形的頂點
polygon = np.array([[380, 2447], [456, 2445], [593, 2709],
                    [904, 3156], [1106, 3325], [1301, 3489],
                    [1639, 3688], [2017, 5446], [799, 5788]], dtype=np.int32)

# 創建一個與原圖大小相同的黑色圖像作為掩膜
mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [polygon], 255)

# 提取 ROI 區域
roi = cv2.bitwise_and(image, image, mask=mask)

# 將提取出的 ROI 區域轉換為灰度圖像
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 對提取出的 ROI 區域進行自適應閾值處理
adaptive_thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# 創建一個與原圖大小相同的黑色圖像作為結果圖像
result = np.zeros_like(gray_roi)

# 將處理後的 ROI 區域放回到原圖相應的位置
result[mask == 255] = adaptive_thresh[mask == 255]

cv2.imshow('Result', cv2.resize(result, None, fx=0.3, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
