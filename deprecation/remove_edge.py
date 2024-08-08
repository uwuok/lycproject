import cv2
import matplotlib.pyplot as plt
def remove_edge() -> int:
    return 0

if __name__ == '__main__':

    # 读入灰度图像
    image = cv2.imread('../area2_2.png', cv2.IMREAD_GRAYSCALE)

    # 检测边缘
    edges = cv2.Canny(image, 700, 400)

    # 找到边缘的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 二值化
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_ori = binary_image
    # 在二值化图像上将轮廓点设为黑色
    g = cv2.drawContours(binary_image, contours, -1, (0), thickness=100)
    cv2.imshow('origin', cv2.resize(image, None, fx=0.3, fy=0.3))
    cv2.imshow('ori_bin', cv2.resize(binary_image_ori, None, fx=0.3, fy=0.3))
    cv2.imshow('remove_edge_binary', cv2.resize(binary_image, None, fx=0.3, fy=0.3))
    cv2.waitKey()
    cv2.destroyAllWindows()
