import cv2

if __name__ == '__main__':
    img = cv2.imread('../area2.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, output1 = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    output2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
    output3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # output3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('origin', cv2.resize(gray_img, None, fx=0.3, fy=0.3))
    cv2.imshow('output1', cv2.resize(output1, None, fx=0.3, fy=0.3))
    cv2.imshow('output2', cv2.resize(output2, None, fx=0.3, fy=0.3))
    cv2.imshow('output3', cv2.resize(output3, None, fx=0.3, fy=0.3))
    cv2.waitKey()
    cv2.destroyAllWindows()
