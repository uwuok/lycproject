import cv2
import schedule
import time
import threading
from datetime import datetime
# import keyboard
import imgpro
import image_algorithm


# 初始化相機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 6000)
cnt = 0
stop_flag = False  # 全局停止旗標

def take_picture():
    if not cap.isOpened() or stop_flag:
        print("停止拍攝或無法抓取鏡頭")
        return

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"photo_{timestamp}.png"

        # 開啟一個新執行緒來保存圖片
        threading.Thread(target=cv2.imwrite, args=(filename, frame)).start()
        threading.Thread(target=imgpro.proc_0919, args=(frame, timestamp)).start()
        global cnt
        cnt += 1
        print(f"已拍攝照片並保存為： {filename} [{cnt}]")
    else:
        print("無法讀取鏡頭 frame")

def schedule_worker():
    while not stop_flag:
        schedule.run_pending()
        time.sleep(0.01)  # 更小的 sleep 間隔以減少等待時間


if __name__ == '__main__':
    # 設定每 8 秒執行一次拍照
    schedule.every(8).seconds.do(take_picture)

    print("程式已啟動，按 Ctrl+C 退出")
    try:
        # 啟動排程檢查執行緒
        threading.Thread(target=schedule_worker).start()

        # 主執行緒等待事件驅動
        while not stop_flag:
            time.sleep(1)  # 保持主執行緒活躍
    except KeyboardInterrupt:
        print("ctrl+c pressed")
        stop_flag = True  # 設置停止旗標
        cap.release()  # 釋放相機資源
        print("程式已結束")
