import numpy as np
import cv2


# not recommended
def canny_method():
    a1 = cv2.imread('final_new_roi1.png', cv2.IMREAD_GRAYSCALE)
    a2 = cv2.imread('final_new_roi2.png', cv2.IMREAD_GRAYSCALE)
    a3 = cv2.imread('final_new_roi3.png', cv2.IMREAD_GRAYSCALE)

    a1 = cv2.bilateralFilter(a1, 5, 50, 100)
    a2 = cv2.bilateralFilter(a2, 5, 50, 100)
    a3 = cv2.bilateralFilter(a3, 5, 50, 100)

    e1 = cv2.Canny(a1, 350, 1000)
    e2 = cv2.Canny(a2, 350, 1000)
    e3 = cv2.Canny(a3, 350, 1000)

    cv2.imwrite('roi1_edge.png', e1)
    cv2.imwrite('roi2_edge.png', e2)
    cv2.imwrite('roi3_edge.png', e3)


def drawpoly(img, points):
    cv2.polylines(img, [np.array(points)], isClosed=True, color=(255, 255, 255), thickness=10)
    return img


if __name__ == '__main__':
    p1 = np.array(
        [[2, 22], [910, 126], [750, 238], [666, 306], [522, 442], [438, 538], [378, 618], [342, 674], [310, 722],
         [270, 794], [246, 846], [202, 934], [174, 1026], [150, 1110], [138, 1162], [126, 1242], [114, 1342],
         [110, 1382], [98, 1382]])
    p2 = np.array([[5, 15], [585, 25], [615, 105], [645, 185], [675, 265], [715, 335], [765, 405], [815, 485], [875, 565], [935, 655], [975, 695], [1015, 745], [1075, 805], [1115, 845], [1125, 865], [1165, 895], [1205, 935], [1255, 975], [1295, 1025], [1335, 1045], [1365, 1075], [1415, 1115], [1445, 1135], [1485, 1175], [1545, 1225], [1575, 1235], [1635, 1275], [1655, 1295], [1695, 1325], [1725, 1345], [1745, 1355], [1795, 1385], [2045, 2995], [125, 3115]])
    p3 = np.array([[275, 430], [3485, 0], [4595, 1820], [425, 3185], [125, 1855], [45, 1515], [5, 1195], [15, 995], [75, 735]])

    a1 = cv2.imread('final_new_roi1.png', cv2.IMREAD_GRAYSCALE)
    a2 = cv2.imread('final_new_roi2.png', cv2.IMREAD_GRAYSCALE)
    a3 = cv2.imread('final_new_roi3.png', cv2.IMREAD_GRAYSCALE)
    m1 = np.zeros_like(a1)
    m2 = np.zeros_like(a2)
    m3 = np.zeros_like(a3)
    m1 = drawpoly(m1, p1)
    m2 = drawpoly(m2, p2)
    m3 = drawpoly(m3, p3)
    cv2.imwrite('roi1_edge.png', m1)
    cv2.imwrite('roi2_edge.png', m2)
    cv2.imwrite('roi3_edge.png', m3)
