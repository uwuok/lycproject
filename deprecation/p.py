import cv2
import schedule
import time
from datetime import datetime
import threading

cnt = 0


def worker():
    print("程式已啟動，按 Ctrl+C 退出")
    try:
        while True:
            schedule.run_pending()
            # time.sleep(1)
    except KeyboardInterrupt:
        print("程式已结束")

def f():
    global cnt
    cnt += 1
    print(f'ABC[{cnt}]')

if __name__ == '__main__':

    # 每5秒执行一次take_picture函数
    schedule.every(2).seconds.do(f)
    # schedule.run_pending()
    threading.Thread(target=worker())
    # worker()
    # cap.release()
