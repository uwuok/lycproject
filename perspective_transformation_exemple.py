import cv2
import numpy as np

def calculate_aspect_ratio(pts):
    # 計算四個頂點所構成四邊形的邊長
    width1 = np.linalg.norm(pts[0] - pts[1])
    width2 = np.linalg.norm(pts[2] - pts[3])
    height1 = np.linalg.norm(pts[0] - pts[3])
    height2 = np.linalg.norm(pts[1] - pts[2])

    # 計算平均寬度和高度
    width = (width1 + width2) / 2
    height = (height1 + height2) / 2

    return width, height

if __name__ == '__main__':
    # 定義特徵點
    uv = np.array([[1208, 1192], [2814, 1116], [3557, 1866], [1160, 2089]], dtype=np.float32)

    # 計算變換後圖像的寬高比例
    # width, height = calculate_aspect_ratio(uv)
    width = 297 * 4
    height = 210 * 4

    # 設置變換後的目標點
    xy = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # 讀取圖像
    image = cv2.imread('side.jpg')

    # 計算透視變換矩陣
    transform_matrix = cv2.getPerspectiveTransform(uv, xy)

    # 應用透視變換
    output_size = (int(width), int(height))
    transformed_image = cv2.warpPerspective(image, transform_matrix, output_size)

    # 顯示結果
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
