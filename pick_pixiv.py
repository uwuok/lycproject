import cv2

WIN_NAME = 'pick_points'
# IMAGE_FILE = '../opencv/color.jpg'
IMAGE_FILE = '/new2.png'


def pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d' % (x, y))


if __name__ == '__main__':
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, pick_points)
    image = cv2.imread(IMAGE_FILE)
    while True:
        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cv2.destroyAllWindows()

