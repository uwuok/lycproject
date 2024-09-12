import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

ms1 = np.array([[7, 38], [69, 207], [146, 376], [176, 438], [253, 546],
                [330, 653], [376, 730], [438, 807], [484, 869], [546, 930],
                [607, 992], [684, 1053], [761, 1130], [823, 1176], [884, 1223],
                [992, 1315], [1053, 1346], [1130, 1392], [1192, 1423],
                [1238, 1453], [1238, 1576], [1269, 1807], [1300, 2084],
                [1315, 2207], [1346, 2376], [1469, 3238], [346, 3469]])

ms2 = np.array([[0, 0], [538, 0], [894, 74], [798, 134],
                [766, 158], [698, 202], [622, 266], [578, 310],
                [522, 362], [426, 478], [374, 550], [318, 626],
                [294, 674], [254, 758], [210, 842], [182, 910],
                [158, 982], [134, 1078], [130, 1114], [22, 1114],
                [0, 974]])


# ms2 = np.array([[0, 0], [950, 60], [882, 104], [818, 144], [734, 196],
#                 [674, 244], [598, 308], [530, 380], [458, 460], [410, 520],
#                 [358, 592], [326, 644], [282, 716], [250, 780], [230, 828],
#                 [202, 892], [182, 936], [162, 1000], [154, 1040], [138, 1096],
#                 [134, 1116], [78, 1116]])


def mask_roi(img, ps):
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    # cv2.imshow('mask', cv2.resize(mask, None, fx=0.2, fy=0.2))
    # cv2.waitKey()
    roi = cv2.bitwise_and(img, img, mask=mask)
    # x1, y1, w1, h1 = cv2.boundingRect(ps1)
    # x2, y2, w2, h2 = cv2.boundingRect(ps2)
    # roi1_crop = roi1[y1:y1 + h1, x1:x1 + w1]
    # roi2_crop = roi2[y2:y2 + h2, x2:x2 + w2]
    # total_area = img.shape[0] * img.shape[1]
    # roi_area = cv2.contourArea(ps)

    # print(roi.shape[0] * roi.shape[1])
    # print(cv2.countNonZero(mask))
    # print(cv2.contourArea(ps))
    # cv2.imshow('img', cv2.resize(img, None, fx=0.2, fy=0.2))
    # sorted_roi = np.sort(roi.flatten())
    # sorted_roi = sorted_roi.reshape(roi.shape)
    # sorted_mask = np.sort(mask.flatten())
    # sorted_mask = sorted_mask.reshape(mask.shape)
    # cv2.imshow('roi', cv2.resize(roi, None, fx=0.2, fy=0.2))
    # cv2.imshow('sorted roi', cv2.resize(sorted_roi, None, fx=0.2, fy=0.2))
    # cv2.imshow('mask', cv2.resize(mask, None, fx=0.2, fy=0.2))
    # cv2.imshow('sorted mask', cv2.resize(sorted_mask, None, fx=0.2, fy=0.2))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    roi_area = cv2.contourArea(ps)
    white_pixels = cv2.countNonZero(roi)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area
    #
    print(f"ROI 面積: {roi_area} 像素")
    print(f"ROI 白色像素： {white_pixels}")
    print(f"ROI 黑色像素： {black_pixels}")
    print(f"ROI 白色像素比例: {white_ratio:.2%}")
    print(f"ROI 黑色像素比例: {black_ratio:.2%}")

    return roi


def get_roi(image):
    # image = cv2.imread(r"C:\Users\natsumi\PycharmProjects\pythonProject\image\photo_20240718_153508.png")
    # x, y
    pts1 = np.array([[560, 2539], [1550, 2504], [2053, 5699], [880, 5982]], np.int32)
    pts2 = np.array([[370, 410], [1341, 466], [1323, 1515], [488, 1528]], np.int32)

    # 最小邊界矩形並擷取 ROI
    x1, y1, w1, h1 = cv2.boundingRect(pts1)
    roi1 = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = cv2.boundingRect(pts2)
    roi2 = image[y2:y2 + h2, x2:x2 + w2]
    print(f'pts1 = ({x1}, {y1}), ({x1}, {y1 + h1}), ({x1 + w1}, {y1 + h1}), ({x1 + w1}, {y1})')
    print(f'pts2 = ({x2}, {y2}), ({x2}, {y2 + h2}), ({x2 + w2}, {y2 + h2}), ({x2 + w2}, {y2})')
    # print(f'pts1 = ({y1}, {x1}), ({y1}, {x1 + w1}), ({y1 + h1}, {x1 + w1}), ({y1 + h1}, {x1})')
    # print(f'pts2 = ({y2}, {x2}), ({y2}, {x2 + w2}), ({y2 + h2}, {x2 + w2}), ({y2 + h2}, {x2})')
    # print(f'pts1 = [[0, 0], [0, {w1}], [{w1}, {h1}], [{h1}, 0]]')
    # print(f'pts1 = [[0, 0], [0, {w2}], [{w2}, {h2}], [{h2}, 0]]')
    # cv2.imwrite(r'roi_output_1.jpg', roi1)
    # cv2.imwrite(r'roi_output_2.jpg', roi2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return roi1, roi2


def image_processed(image):
    # 讀取原始圖像
    # image = cv2.imread(r"C:\Users\User\Desktop\roi_output2.jpg")
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 高斯模糊 (降噪)
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    # cv2.imshow('blurred_image', cv2.resize(blurred_image, None, fx=0.2, fy=0.2))
    # 銳化處理 (利用 PIL 提升對比度)
    enhancer = ImageEnhance.Contrast(Image.fromarray(blurred_image))
    sharp_img = enhancer.enhance(2)  # 銳化系數
    sharp_img_np = np.array(sharp_img)  # 轉回 NumPy 格式
    # cv2.imshow('sharp_img.png', cv2.resize(sharp_img_np, None, fx=0.2, fy=0.2))
    # 限制對比度自適應直方圖均衡化
    gray_img = cv2.cvtColor(sharp_img_np, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', cv2.resize(gray_img, None, fx=0.2, fy=0.2))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    clahe_img = clahe.apply(gray_img)
    # cv2.imshow('clahe', cv2.resize(clahe_img, None, fx=0.2, fy=0.2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    tophat_img = cv2.morphologyEx(clahe_img, cv2.MORPH_TOPHAT, kernel)
    # cv2.imshow('top hat', cv2.resize(tophat_img, None, fx=0.2, fy=0.2))
    threshold = cv2.adaptiveThreshold(tophat_img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -8)
    # cv2.imshow('adaptive threshold', cv2.resize(threshold, None, fx=0.2, fy=0.2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # er = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=1)
    re = cv2.dilate(threshold, kernel, iterations=7)
    # cv2.imshow('dilate', cv2.resize(re, None, fx=0.2, fy=0.2))
    opening = cv2.morphologyEx(re, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow('opening', cv2.resize(opening, None, fx=0.2, fy=0.2))
    # plt.subplot(1, 5, 1)
    # plt.imshow(image_rgb)
    # plt.title('Original Image')
    # plt.axis('off')
    # plt.subplot(1, 5, 2)
    # plt.imshow(clahe_img, cmap='gray')
    # plt.title('CLAHE Image')
    # plt.axis('off')
    # plt.subplot(1, 5, 3)
    # plt.imshow(tophat_img, cmap='gray')
    # plt.title('Top-hat Image')
    # plt.axis('off')
    # plt.subplot(1, 5, 4)
    # plt.imshow(threshold, cmap='gray')
    # plt.title('thr')
    # plt.axis('off')
    # plt.subplot(1, 5, 5)
    # plt.imshow(opening, cmap='gray')
    # plt.title('re')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
    return opening

def v2(image):
    # 銳化處理 (利用 PIL 提升對比度)
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    sharp_img = enhancer.enhance(2)  # 銳化系數
    sharp_img_np = np.array(sharp_img)  # 轉回 NumPy 格式
    # 灰階
    gray = cv2.cvtColor(sharp_img_np, cv2.COLOR_BGR2GRAY)
    # 二值化
    # b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 77, -1)
    b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 777, -1)
    b = cv2.bitwise_not(b)
    return b


if __name__ == '__main__':
    image = cv2.imread('sample.png')
    # 整體亮度 117(11)，判斷積屑量過少
    # image = cv2.imread('photo_20240718_154734.png')

    # 整體亮度低 9，判斷積屑量 OK
    # image = cv2.imread('photo_20240718_162814.png')
    # 整體亮度高 18，判斷積屑量過多
    # image = cv2.imread('photo_20240718_153508.png')

    roi1, roi2 = get_roi(image)
    # cv2.imshow('roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    # cv2.imshow('roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    proc_roi1 = v2(roi1)
    proc_roi2 = v2(roi2)
    # cv2.imshow('proc_roi1', cv2.resize(proc_roi1, None, fx=0.2, fy=0.2))
    # cv2.imshow('proc_roi2', cv2.resize(proc_roi2, None, fx=0.2, fy=0.2))
    res1 = mask_roi(proc_roi1, ms1)
    res2 = mask_roi(proc_roi2, ms2)
    cv2.imshow('1', cv2.resize(res1, None, fx=0.2, fy=0.2))
    cv2.imshow('2', cv2.resize(res2, None, fx=0.2, fy=0.2))
    # cv2.imwrite('1.png', res1)
    # cv2.imwrite('2.png', res2)
    cv2.waitKey()
    cv2.destroyAllWindows()
