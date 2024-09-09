import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

points = []


def mask_roi(roi1, roi2):
    ps1 = np.array([[7, 38],
                    [69, 207],
                    [146, 376],
                    [176, 438],
                    [253, 546],
                    [330, 653],
                    [376, 730],
                    [438, 807],
                    [484, 869],
                    [546, 930],
                    [607, 992],
                    [684, 1053],
                    [761, 1130],
                    [823, 1176],
                    [884, 1223],
                    [992, 1315],
                    [1053, 1346],
                    [1130, 1392],
                    [1192, 1423],
                    [1238, 1453],
                    [1238, 1576],
                    [1269, 1807],
                    [1300, 2084],
                    [1315, 2207],
                    [1346, 2376],
                    [1469, 3238],
                    [346, 3469]])
    ps2 = np.array([[0, 0],
                    [950, 60],
                    [882, 104],
                    [818, 144],
                    [734, 196],
                    [674, 244],
                    [598, 308],
                    [530, 380],
                    [458, 460],
                    [410, 520],
                    [358, 592],
                    [326, 644],
                    [282, 716],
                    [250, 780],
                    [230, 828],
                    [202, 892],
                    [182, 936],
                    [162, 1000],
                    [154, 1040],
                    [138, 1096],
                    [134, 1116],
                    [78, 1116]])
    mask_shape = roi1.shape[:2]
    m1 = np.zeros(mask_shape, dtype=np.uint8)
    mask_shape = roi2.shape[:2]
    m2 = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(m1, [ps1], 255)
    cv2.fillPoly(m2, [ps2], 255)
    roi1 = cv2.bitwise_and(roi1, roi1, mask=m1)
    roi2 = cv2.bitwise_and(roi2, roi2, mask=m2)
    # x1, y1, w1, h1 = cv2.boundingRect(ps1)
    # x2, y2, w2, h2 = cv2.boundingRect(ps2)
    # roi1_crop = roi1[y1:y1 + h1, x1:x1 + w1]
    # roi2_crop = roi2[y2:y2 + h2, x2:x2 + w2]


    roi_area = cv2.contourArea(ps1)
    white_pixels = cv2.countNonZero(roi1)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    print(f"ROI1 面積: {roi_area} 像素")
    print(f"ROI1 白色像素： {white_pixels}")
    print(f"ROI1 黑色像素： {black_pixels}")
    print(f"ROI1 白色像素比例: {white_ratio:.2%}")
    print(f"ROI1 黑色像素比例: {black_ratio:.2%}")


    roi_area = cv2.contourArea(ps2)
    white_pixels = cv2.countNonZero(roi2)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area

    print(f"ROI2 面積: {roi_area} 像素")
    print(f"ROI2 白色像素： {white_pixels}")
    print(f"ROI2 黑色像素： {black_pixels}")
    print(f"ROI2 白色像素比例: {white_ratio:.2%}")
    print(f"ROI2 黑色像素比例: {black_ratio:.2%}")

    return roi1, roi2


def get_roi(image):
    # image = cv2.imread(r"C:\Users\natsumi\PycharmProjects\pythonProject\image\photo_20240718_153508.png")

    pts1 = np.array([[560, 2539], [1550, 2504], [
        2053, 5699], [880, 5982]], np.int32)

    pts2 = np.array([[370, 410], [1341, 466], [1323, 1515], [488, 1528]], np.int32)

    # 最小邊界矩形並擷取 ROI
    x1, y1, w1, h1 = cv2.boundingRect(pts1)
    roi1 = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = cv2.boundingRect(pts2)
    roi2 = image[y2:y2 + h2, x2:x2 + w2]
    print(f'pts1 = [[0, 0], [0, {w1}], [{w1}, {h1}], [{h1}, 0]]')
    print(f'pts1 = [[0, 0], [0, {w2}], [{w2}, {h2}], [{h2}, 0]]')
    cv2.imwrite(r'roi_output_1.jpg', roi1)
    cv2.imwrite(r'roi_output_2.jpg', roi2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return roi1, roi2


def image_processed(image):
    # 讀取原始圖像
    # image = cv2.imread(r"C:\Users\User\Desktop\roi_output2.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 高斯模糊 (降噪)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # 銳化處理 (利用 PIL 提升對比度)
    enhancer = ImageEnhance.Contrast(Image.fromarray(blurred_image))
    sharp_img = enhancer.enhance(2)  # 銳化系數
    sharp_img_np = np.array(sharp_img)  # 轉回 NumPy 格式
    # 限制對比度自適應直方圖均衡化
    gray_img = cv2.cvtColor(sharp_img_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    filterSize = (8, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    tophat_img = cv2.morphologyEx(clahe_img, cv2.MORPH_TOPHAT, kernel)
    threshold = cv2.adaptiveThreshold(
        tophat_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    re = cv2.dilate(threshold, kernel, iterations=4)
    opening = cv2.morphologyEx(re, cv2.MORPH_OPEN, kernel, iterations=2)
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
    return opening


if __name__ == '__main__':
    # image = cv2.imread('photo_20240718_154734.png')
    image = cv2.imread('new2.png')
    roi1, roi2 = get_roi(image)
    cv2.imshow('roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    proc_roi1 = image_processed(roi1)
    proc_roi2 = image_processed(roi2)
    cv2.imshow('proc_roi1', cv2.resize(proc_roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('proc_roi2', cv2.resize(proc_roi2, None, fx=0.2, fy=0.2))
    res1, res2 = mask_roi(proc_roi1, proc_roi2)
    cv2.imshow('1', cv2.resize(res1, None, fx=0.2, fy=0.2))
    cv2.imshow('2', cv2.resize(res2, None, fx=0.2, fy=0.2))
    cv2.waitKey()
    cv2.destroyAllWindows()
