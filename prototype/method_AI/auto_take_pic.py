import cv2
import schedule
import time
import threading
from datetime import datetime
# import keyboard
import img_pre_proc
import os
import call_module
import json
import numpy as np 

# 初始化相機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 6000)
cnt = 0
stop_flag = False  # 全局停止旗標
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

def update_json(value):
    file_path = r'D:\prototype_test\method_AI\config.json'
    
    # 檢查文件是否存在，如果不存在則創建一個空的 JSON 文件
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({}, file)
    
    try:
        # 讀取現有的 JSON 數據
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # 更新正確的 key
        data['Flusher_level_bar'] = 0.0
        data['Flusher_level'] = value
        
        # 將更新的內容寫入到 JSON 文件
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        

import concurrent.futures

def take_picture():
    if not cap.isOpened() or stop_flag:
        print("停止拍攝或無法抓取鏡頭")
        return

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"photo_{timestamp}.png"
        threading.Thread(target=cv2.imwrite, args=(filename, frame)).start()

        # 使用執行緒池處理預處理與 AI 模塊
        with concurrent.futures.ThreadPoolExecutor() as executor:
            r1_future = executor.submit(img_pre_proc.pre_proc, frame)
            r1 = r1_future.result()  # 等待結果
            
            # Assuming r1 is a tuple or list, extract the actual image if it's the first element
            # Change this depending on the actual structure of `r1`
            if isinstance(r1, (tuple, list)):
                r1 = r1[0]  # Extract the first element if r1 is a tuple/list

            # Ensure r1 is a proper NumPy array before passing
            r1 = np.array(r1)
            r1 = r1.astype(np.float64)

            predict_result_future = executor.submit(call_module.single_file, r1)
            predict_result = predict_result_future.result()  # 等待 AI 結果
            predict_result = int(predict_result)

        # 寫入到 JSON
        print(f'predict_result = {predict_result}')
        update_json(predict_result)

        global cnt
        cnt += 1
        print(f"已拍攝照片並保存為： {filename} [{cnt}]\n")
    else:
        print("無法讀取鏡頭 frame\n")



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
