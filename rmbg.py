import cv2
import numpy as np

# 滑鼠回調函數
points = []
scale = 0.5  # 縮小比例

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()


# [(4, 0), (852, 30), (430, 420), (308, 608), (236, 756), (180, 914), (144, 1050), (112, 1208)]

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

    print(scaled_points)

    # 創建掩膜
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if scaled_points:
        cv2.fillPoly(mask, [np.array(scaled_points)], color=255)

    return mask

if __name__ == '__main__':
    img = cv2.imread('roiroiroinnn.png')

    # 縮放比例
    scale = 0.5  # 根據需要調整縮放比例

    mask = getPolygonROI(img, scale)

    b_Model = np.zeros((1, 65), np.float64)
    f_Model = np.zeros((1, 65), np.float64)

    # 初始化掩膜：0 - 背景, 1 - 前景, 2 - 可能的背景, 3 - 可能的前景
    mask[mask == 255] = 3  # 設定多邊形內部為可能的前景
    mask[mask == 0] = 2  # 其他區域為可能的背景

    cv2.grabCut(img, mask, None, b_Model, f_Model, 5, cv2.GC_INIT_WITH_MASK)

    # 提取前景
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_cut = img * mask2[:, :, np.newaxis]

    # print(points)

    cv2.imshow('Cut Image', img_cut)
    cv2.imwrite('cut_img.png', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
