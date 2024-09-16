import cv2

# 读取图像
image = cv2.imread('../new.png')
template = cv2.imread('dst2.jpg')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 阈值化处理
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
_, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours_image, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_template, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算轮廓特征（Hu 矩）
moments_image = cv2.moments(contours_image[0])
moments_template = cv2.moments(contours_template[0])
hu_image = cv2.HuMoments(moments_image).flatten()
hu_template = cv2.HuMoments(moments_template).flatten()

# 计算相似度
similarity = cv2.matchShapes(contours_image[0], contours_template[0], cv2.CONTOURS_MATCH_I1, 0.0)
print("Shape Similarity:", similarity)

# 显示结果
cv2.drawContours(image, contours_image, -1, (0, 255, 0), 2)
cv2.drawContours(template, contours_template, -1, (0, 255, 0), 2)
cv2.imshow('Image Contours', cv2.resize(image, None, fx=0.1, fy=0.1))
cv2.imshow('Template Contours', cv2.resize(template, None, fx=0.1, fy=0.1))
cv2.waitKey(0)
cv2.destroyAllWindows()
