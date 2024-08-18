import cv2
import numpy as np

# 定義點的座標
# points = np.array([[336, 484], [1296, 515], [1333, 1621], [442, 1698]], dtype=np.int32)
# points = np.array([[336, 484], [1296, 515], [442, 1698]], dtype=np.int32)
# points = np.array([(0, 2520), (560, 2540), (680, 2800), (820, 3040), (1070, 3390), (1240, 3560), (1610, 3810), (1780, 3930), (2070, 5460), (120, 5710)])
# points = points.reshape((-1, 1, 2))

points = []


def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()


def getPolygonROI(img, scale):
    global points
    points = []

    # 縮小圖像
    resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # 顯示圖像並設置滑鼠回調
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_polygon)

    while True:
        temp_img = resized_img.copy()
        if points:
            cv2.polylines(temp_img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Image", temp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # 按 'q' 完成選取
            break

    cv2.destroyAllWindows()

    # 將選取的多邊形座標放大回原始尺寸
    scaled_points = [(int(x / scale), int(y / scale)) for x, y in points]
    scaled_points = np.array(scaled_points, dtype=np.int32)  # 確保轉換為 NumPy 陣列

    print(scaled_points)
    return scaled_points


if __name__ == '__main__':
    img = cv2.imread('new.png', cv2.IMREAD_UNCHANGED)

    # 創建黑色遮罩
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # 填充多邊形範圍
    ps = getPolygonROI(img, scale=0.1)
    cv2.fillPoly(mask, [ps], 255)

    # 提取 ROI
    roi = cv2.bitwise_and(img, img, mask=mask)

    # 計算邊界矩形
    x, y, w, h = cv2.boundingRect(ps)  # 使用縮放後的點來計算邊界
    roi_crop = roi[y:y+h, x:x+w]

    # 顯示結果
    cv2.imshow('ROI Crop', cv2.resize(roi_crop, None, fx=0.2, fy=0.2))
    cv2.imwrite('roi2.png', roi_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()
