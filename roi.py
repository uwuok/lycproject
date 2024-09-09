import cv2
import numpy as np
image = cv2.imread(r"C:\Users\natsumi\PycharmProjects\pythonProject\image\photo_20240718_153508.png")
pts1 = np.array([[560, 2539], [1550, 2504], [
    2053, 5699], [880, 5982]], np.int32)

pts2 = np.array([[370, 410], [1341, 466], [1323, 1515], [488, 1528]], np.int32)

# 最小邊界矩形並擷取 ROI
x1, y1, w1, h1 = cv2.boundingRect(pts1)
roi1 = image[y1:y1+h1, x1:x1+w1]
x2, y2, w2, h2 = cv2.boundingRect(pts2)
roi2 = image[y2:y2+h2, x2:x2+w2]
print(f'pts1 = [[0, 0], [0, {w1}], [{w1}, {h1}], [{h1}, 0]]')
print(f'pts1 = [[0, 0], [0, {w2}], [{w2}, {h2}], [{h2}, 0]]')
cv2.imwrite(r'roi_output_1.jpg', roi1)
cv2.imwrite(r'roi_output_2.jpg', roi2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()