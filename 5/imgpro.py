import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance

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


def mask_roi(img, ps):
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    # cv2.imshow('mask', cv2.resize(mask, None, fx=0.2, fy=0.2))
    # cv2.waitKey()
    roi_area = np.count_nonzero(mask)
    roi = cv2.bitwise_and(img, img, mask=mask)
    if len(roi.shape) > 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white_pixels = cv2.countNonZero(roi)
    black_pixels = roi_area - white_pixels
    white_ratio = white_pixels / roi_area
    black_ratio = black_pixels / roi_area
    #
    print(f"ROI 面積: {roi_area} 像素")
    # print(f"ROI 白色像素： {white_pixels}")
    # print(f"ROI 黑色像素： {black_pixels}")
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


def test_ppt():
    cnt = 0
    current_path = os.getcwd()  # 獲取當前資料夾路徑
    files = os.listdir(current_path)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')

    for file in files:
        if file.lower().endswith(image_extensions):
            cnt += 1
            print(f'已處理圖片數：{cnt}')
            print(f'當前圖片名稱：{file}')
            current_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            roi1, roi2 = get_roi(current_img)  # 獲取兩個 ROI 區域

            # 創建每張圖片的唯一保存名稱，防止名稱重複
            base_filename = os.path.splitext(file)[0]

            fx, fy = 0.2, 0.2

            # 顯示和保存第一個 ROI
            cv2.imshow('ori_img_1', cv2.resize(roi1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_ori_img_1.jpg'), cv2.resize(roi1, None, fx=fx, fy=fy))

            # 顯示和保存第二個 ROI
            cv2.imshow('ori_img_2', cv2.resize(roi2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_ori_img_2.jpg'), cv2.resize(roi2, None, fx=fx, fy=fy))

            # 灰階處理
            g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray_img_1', cv2.resize(g1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_gray_img_1.jpg'), cv2.resize(g1, None, fx=fx, fy=fy))
            cv2.imshow('gray_img_2', cv2.resize(g2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_gray_img_2.jpg'), cv2.resize(g2, None, fx=fx, fy=fy))

            # 銳化處理
            s1 = sharp(g1)
            s2 = sharp(g2)
            cv2.imshow('sharp_img_1', cv2.resize(s1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_sharp_img_1.jpg'), cv2.resize(s1, None, fx=fx, fy=fy))
            cv2.imshow('sharp_img_2', cv2.resize(s2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_sharp_img_2.jpg'), cv2.resize(s2, None, fx=fx, fy=fy))

            # 邊緣檢測
            # 15, 28
            e1 = cv2.Canny(s1, 25, 50, L2gradient=True)
            e2 = cv2.Canny(s2, 25, 50, L2gradient=True)
            cv2.imshow('edges_1', cv2.resize(e1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_edges_1.jpg'), cv2.resize(e1, None, fx=fx, fy=fy))
            cv2.imshow('edges_2', cv2.resize(e2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_edges_2.jpg'), cv2.resize(e2, None, fx=fx, fy=fy))

            # 膨脹處理
            r1 = cv2.dilate(e1, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
            r2 = cv2.dilate(e2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
            cv2.imshow('dilate_1', cv2.resize(r1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_dilate_1.jpg'), cv2.resize(r1, None, fx=fx, fy=fy))
            cv2.imshow('dilate_2', cv2.resize(r2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_dilate_2.jpg'), cv2.resize(r2, None, fx=fx, fy=fy))

            # 掩膜處理
            r1 = mask_roi(r1, ms1)
            r2 = mask_roi(r2, ms2)
            cv2.imshow('mask_roi_1', cv2.resize(r1, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_mask_roi_1.jpg'), cv2.resize(r1, None, fx=fx, fy=fy))
            cv2.imshow('mask_roi_2', cv2.resize(r2, None, fx=fx, fy=fy))
            cv2.imwrite(os.path.join(current_path, f'{base_filename}_mask_roi_2.jpg'), cv2.resize(r2, None, fx=fx, fy=fy))

            # 等待按鍵和關閉所有視窗
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def sharp(image):
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    sharp_img = enhancer.enhance(2)
    sharp_img_np = np.array(sharp_img)
    return sharp_img_np


def v4(img, points):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhancer = ImageEnhance.Contrast(Image.fromarray(gray_img))
    sharp_img = enhancer.enhance(2)  # 銳化系數
    sharp_img_np = np.array(sharp_img)  # 轉回 NumPy 格式
    # 10 為弱邊緣，150 為強邊緣
    edges = cv2.Canny(sharp_img_np, 10, 13, L2gradient=True)
    # cv2.imshow('edges', cv2.resize(edges, None, fx=0.2, fy=0.2))
    # 邊界跟蹤
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_area = 5
    # foreground = np.zeros_like(edges)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area >= min_area:
    #         # cv2.drawContours(foreground, [contour], -1, (255, 255, 255), -1)
    #         cv2.drawContours(foreground, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.imshow('foreground', cv2.resize(foreground, None, fx=0.2, fy=0.2))
    # res = mask_roi(foreground, points)
    # cv2.imshow('res', cv2.resize(res, None, fx=0.2, fy=0.2))
    res = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
    res = mask_roi(res, points)
    # cv2.imshow('res', cv2.resize(res, None, fx=0.2, fy=0.2))
    cv2.waitKey()
    cv2.destroyAllWindows()
    return res


def test_v4():
    image = cv2.imread('fail.png')
    roi1, roi2 = get_roi(image)
    cv2.imshow('ori_roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('ori_roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    v4(roi1, ms1)
    v4(roi2, ms2)

    image = cv2.imread('sample_blur.png')
    roi1, roi2 = get_roi(image)
    cv2.imshow('ori_roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('ori_roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    v4(roi1, ms1)
    v4(roi2, ms2)

    image = cv2.imread('sample.png')
    roi1, roi2 = get_roi(image)
    cv2.imshow('ori_roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('ori_roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    v4(roi1, ms1)
    v4(roi2, ms2)

    image = cv2.imread('new2.png')
    roi1, roi2 = get_roi(image)
    cv2.imshow('ori_roi1', cv2.resize(roi1, None, fx=0.2, fy=0.2))
    cv2.imshow('ori_roi2', cv2.resize(roi2, None, fx=0.2, fy=0.2))
    v4(roi1, ms1)
    v4(roi2, ms2)


if __name__ == '__main__':
    test_ppt()
