import cv2
import numpy as np

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

    # 缩小图像
    resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # 显示图像并设置鼠标回调
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_polygon)

    while True:
        temp_img = resized_img.copy()
        if points:
            cv2.polylines(temp_img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Image", temp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 完成选取
            break
        elif key == ord('z'):  # 按 'z' 撤销上一步
            if points:
                points.pop()

    cv2.destroyAllWindows()

    # 将选取的多边形坐标放大回原始尺寸
    scaled_points = [(int(x / scale), int(y / scale)) for x, y in points]
    scaled_points = np.array(scaled_points, dtype=np.int32)  # 确保转换为 NumPy 数组

    print(scaled_points)
    return scaled_points

if __name__ == '__main__':
    img = cv2.imread('../new2.png', cv2.IMREAD_UNCHANGED)

    # 创建黑色遮罩
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # 填充多边形范围
    ps = getPolygonROI(img, scale=0.1)
    cv2.fillPoly(mask, [ps], 255)

    # 提取 ROI
    roi = cv2.bitwise_and(img, img, mask=mask)

    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(ps)  # 使用缩放后的点来计算边界
    roi_crop = roi[y:y+h, x:x+w]

    # 显示结果
    cv2.imshow('ROI Crop', cv2.resize(roi_crop, None, fx=0.2, fy=0.2))
    cv2.imwrite('../roi2.png', roi_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()