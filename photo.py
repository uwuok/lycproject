import cv2
import schedule
import time
from datetime import datetime

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)  # 使用 Media Foundation 後端
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def take_picture():
    # cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("無法抓取鏡頭")
        return
    # 設置解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 6000)

    ret, frame = cap.read()

    if ret:
        # 获取当前时间戳并格式化为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"photo_{timestamp}.png"
        # 保存照片
        cv2.imwrite(filename, frame)
        print(f"已拍攝照片並保存為： {filename}")
    else:
        print("無法讀取鏡頭 frame")

    # 释放摄像头资源
    # cap.release()


if __name__ == '__main__':

    available_cameras = list_cameras()
    print("可用攝影機編號:", available_cameras)

    if not available_cameras:
        print("未找到可用的攝影機")
        exit()

    # 每5秒执行一次take_picture函数
    schedule.every(3).seconds.do(take_picture)

    print("程式已啟動，按 Ctrl+C 退出")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("程式已结束")
        cap.release()