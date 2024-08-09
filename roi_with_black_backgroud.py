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


if __name__ == '__main__':
    img = cv2.imread('1.png')
    mask_shape = img.shape[:2]
    roi1 = get_roi_pic(img, points_1)
    roi2 = get_roi_pic(img, points_2)
    roi3 = get_roi_pic(img, points_3)
    print(img.shape)
    dst1 = cv2.resize(roi1, dsize=(len(roi1[0]), len(roi1[2])), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('area1', cv2.resize(roi1, None, fx=0.5, fy=0.5))
    dst2 = cv2.resize(roi2, dsize=(roi2.shape[1], roi2.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imshow('area2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    dst3 = cv2.resize(roi3, dsize=roi3.shape[1::-1], interpolation=cv2.INTER_AREA)
    cv2.imshow('area3', cv2.resize(roi3, None, fx=0.2, fy=0.2))
    # contour(dst2, )
    cv2.imwrite('1_1.jpg', dst1)
    cv2.imwrite('1_2.jpg', dst2)
    cv2.imwrite('1_3.jpg', dst3)
    cv2.waitKey()
    cv2.destroyAllWindows()
