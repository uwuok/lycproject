import cv2
import numpy as np

imgnew = cv2.imread('roiroiroinnn.png')
height, width = imgnew.shape[:2]

# 缩小图像
scale = 0.2  # 调整缩放比例
resized_img = cv2.resize(imgnew, (int(width * scale), int(height * scale)))

# 选择ROI
rect1 = cv2.selectROI(resized_img)
cv2.destroyAllWindows()

b_Model = np.zeros((1, 65), np.float64)
f_Model = np.zeros((1, 65), np.float64)

# 计算缩放比例
x, y, w, h = rect1
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

# x1,y1,w1,h1 = rect1 #rect1是一個list
print(x, y, w, h)  #57 54 302 510 (x,y座標 ,w1:寬,h1:高)
mask_new, b_model, f_model = cv2.grabCut(imgnew, None, rect1, b_Model, f_Model, 5, cv2.GC_INIT_WITH_RECT)
#沒有用mask所以第二個參數要填none,不然會拋錯,數字5是指跑5次
# grabcut會把算出來的結果存回去 mask_new、b_Model、f_Model
print(mask_new, b_model, f_model)  #2d array

# 在原图上绘制选定的区域
# roi = imgnew[y:y+h, x:x+w]
# cv2.imshow('roi', roi)
# cv2.waitKey()
# cv2.destroyAllWindows()
