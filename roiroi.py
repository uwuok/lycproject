import cv2
import numpy as np
'''
x = 512, y = 2453
x = 1519, y = 2617
x = 2194, y = 5967
x = 881, y = 5992
'''

# x = 336, y = 484
# x = 1296, y = 515
# x = 1333, y = 1621
# x = 442, y = 1698

if __name__ == '__main__':
    img = cv2.imread('new.png', cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # points = np.array([[512, 2453], [1519, 2617], [2194, 6000], [881, 6000]], dtype=np.int32).reshape((-1, 1, 2))
    points = np.array([[336, 484], [1296, 515], [1333, 1621], [442, 1698]])
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    roi = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(points)
    # black_bg = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
    roi_crop = roi[y:y+h, x:x+w]
    # black_bg[:roi_crop.shape[0], :roi_crop.shape[1]] = roi_crop
    # cv2.imshow('res', cv2.resize(black_bg, None, fx=0.3, fy=0.3))
    cv2.imshow('wer', cv2.resize(roi_crop, None, fx=0.2, fy=0.2))
    # cv2.imshow('erwer', cv2.resize(roi, None, fx=0.2, fy=0.2))
    cv2.waitKey()
    cv2.destroyAllWindows()

