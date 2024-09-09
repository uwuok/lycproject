import cv2
import numpy as np

if __name__ == '__main__':
    ori = cv2.imread('1.png', cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.medianBlur(gray, 5)
    binary_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1)

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #設定一下drawContours的參數
    contours_to_plot= -1 #畫全部
    plotting_color= (0,255,0)#畫綠色框
    thickness= 2
    #開始畫contours
    with_contours = cv2.drawContours(ori, contours, contours_to_plot, plotting_color,thickness)
    # cv2.imshow('contours', with_contours)
    cv2.imshow('binary', cv2.resize(binary_image, None, fx=0.1, fy=0.1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
