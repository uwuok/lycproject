import numpy as np
import cv2


def c():
    # 假設為 ROI 圖上的白點
    points = np.array([[0, 0], [0, 11], [10, 10], [10, 0]])
    # 假設為 ROI 所框出的區域
    points_roi = np.array([[0, 0], [0, 30], [30, 30], [30, 0]])
    # 假設為圖像外接矩形
    black_pic = np.zeros((1000, 1000), dtype=np.uint8)

    cv2.fillPoly(black_pic, [points], 255)

    # 計算 ROI 所框出的區域的面積
    roi_area = cv2.contourArea(points_roi)

    # 計算圖像中出現的白點
    cnt = 0
    for i in range(black_pic.shape[0]):
        for j in range(black_pic.shape[1]):
            if black_pic[i][j] > 0:
                cnt += 1
    white_pixes = cv2.countNonZero(black_pic)

    # 計算 ROI 所框出的區域的黑點
    black_pixes = roi_area - white_pixes

    pic_area = black_pic.shape[0] * black_pic.shape[1]
    print('pic area is:', pic_area)
    print("roi area is:", roi_area)
    print('points area', cv2.contourArea(points))
    print('white_pixes is:', white_pixes)
    print('black_pixes is:', black_pixes)
    print(f'white_pixes ratio is:{white_pixes/roi_area:.2%}')
    print(f'black_pixes ratio is:{black_pixes/roi_area:.2%}')
    print('cnt = ', cnt)
    cv2.imshow('pic', black_pic)
    cv2.waitKey()
    cv2.destroyAllWindows()


print(c())
