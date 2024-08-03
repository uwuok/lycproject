import cv2
import numpy as np

'''
roi1 = (1212, 883, 3)   # (y, x, channel)
roi2 = (3344, 1638, 3)
roi3 = (3160, 4495, 3)
'''

points_1 = np.array([[148, 200], [1030, 248], [766, 441],
                     [627, 589], [490, 780], [404, 949],
                     [325, 1167], [269, 1411]])

points_2 = np.array([[380, 2447], [456, 2445], [593, 2709],
                     [904, 3156], [1106, 3325], [1301, 3489],
                     [1639, 3688], [2017, 5446], [799, 5788]])

points_3 = np.array([[2737, 2997], [5957, 2595], [6948, 4363],
                     [2928, 5754], [2487, 4061], [2454, 3595],
                     [2528, 3261], [2626, 3097]])

def get_bounding_box_corners(points):
    width = int(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = int(np.max(points[:, 1] - np.min(points[:, 1])))
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


def perspective_transform(image, points):
    bounding_box_corners = get_bounding_box_corners(points)
    width = int(np.max(bounding_box_corners[:, 0]) - np.min(bounding_box_corners[:, 0]))
    height = int(np.max(bounding_box_corners[:, 1]) - np.min(bounding_box_corners[:, 1]))
    m = cv2.getPerspectiveTransform(points, bounding_box_corners)
    result = cv2.warpPerspective(image, m, (width, height))
    return result

# def perspective_transform(image):
#     width = image.shape[0]
#     height = image.shape[1]
#     bounding_box_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
#     m = cv2.getPerspectiveTransform(bounding_box_corners, bounding_box_corners)
#     result = cv2.warpPerspective(image, m, (width, height))
#     return result

'''
def perspective_transform(image, points, new_points=get_bounding_box_corners()):
    bounding_box_corners = new_points
    width = int(np.max(bounding_box_corners[:, 0]) - np.min(bounding_box_corners[:, 0]))
    height = int(np.max(bounding_box_corners[:, 1]) - np.min(bounding_box_corners[:, 1]))
    m = cv2.getPerspectiveTransform(points, new_points)
    result = cv2.warpPerspective(image, m, (width, height))
    return result
'''

def draw_polygon(image, points, color, thickness):
    points = points.astype(np.int32)        # convert points to int32
    points2 = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)


# def get_roi_pic(image, points: DEFAULT_DTYPE, mask_shape):
#     points_ = np.array(points, dtype=DEFAULT_DTYPE).reshape((-1, 1, 2))
#     mask = np.zeros(mask_shape, dtype=np.uint8)
#     cv2.fillPoly(mask, [points_], 255)
#     roi_img = cv2.bitwise_and(image, image, mask=mask)
#
#     # 計算最小外接矩形
#     rect = cv2.minAreaRect(points_)
#     box = cv2.boxPoints(rect)
#     box = np.intp(box)
#
#     # 獲取矩形的寬度和高度
#     width = int(rect[1][0])
#     height = int(rect[1][1])
#
#     # 創建黑色背景，大小為最小外接矩形
#     black_bg = np.zeros((height, width, 3), dtype=np.uint8)
#
#     # 計算透視變換矩陣
#     src_pts = np.array(points, dtype=np.float32)
#     dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#
#     # 應用透視變換
#     result = cv2.warpPerspective(roi_img, M, (width, height))
#
#     return result


def get_roi_pic(image, points):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask_shape = img.shape[:2]  # 原圖大小
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


def get_roi_pic_v2(image, points):

    # 將點轉換為 numpy 數組
    points = np.array(points, dtype="float32")

    # 計算最小包圍矩形
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 獲取最小矩形的寬和高
    width = int(rect[1][0])
    height = int(rect[1][1])

    # 定義目標點 (標準矩形的四個頂點)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)

    # 進行透視變換
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped



if __name__ == '__main__':
    img = cv2.imread('2.png')
    roi2 = get_roi_pic_v2(img, points_2)
    cv2.imshow('roi2', cv2.resize(roi2, None, fx=0.3, fy=0.3))
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    # area1 [137, 219], [1027, 246], [1105, 1380], [270, 1447]
    img = cv2.imread('2.png')
    # points1 = np.array([[137, 219], [1027, 246], [1105, 1380], [270, 1447]], dtype=np.float32)
    # new_points1 = np.array([[0, 0], [300, 0], [300, 400], [0, 400]], dtype=np.float32)
    # dst1 = perspective_transform(img, points1)
    # draw_polygon(img, points1, color=(0, 255, 0), thickness=10)
    # cv2.imshow('input', cv2.resize(img, None, fx=0.1, fy=0.1))
    # cv2.imshow('area 1', cv2.resize(dst1, None, fx=0.5, fy=0.5))
    mask_shape = img.shape[:2]
    roi1 = get_roi_pic(img, points_1)
    roi2 = get_roi_pic(img, points_2)
    roi3 = get_roi_pic(img, points_3)

    # p_roi1 = perspective_transform(roi1)
    # p_roi2 = perspective_transform(roi2)
    # p_roi3 = perspective_transform(roi3)
    cv2.imshow('area1', cv2.resize(roi1, None, fx=0.5, fy=0.5))
    # cv2.imshow('area1', cv2.resize(p_roi1, None, fx=0.5, fy=0.5))
    cv2.imshow('area2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    # cv2.imshow('area2', cv2.resize(p_roi2, None, fx=0.2, fy=0.2))
    cv2.imshow('area3', cv2.resize(roi3, None, fx=0.2, fy=0.2))
    # cv2.imshow('area3', cv2.resize(p_roi3, None, fx=0.2, fy=0.2))
    # cv2.imwrite('')
    # cv2.imwrite('area2_2.png', roi2)
    print(f'roi1 = {roi1.shape}')  # y, x, channel
    print(f'roi2 = {roi2.shape}')
    print(f'roi3 = {roi3.shape}')
    cv2.waitKey()
    cv2.destroyAllWindows()
