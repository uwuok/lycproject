from datetime import datetime
import img_pre_proc
import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import time

# 初始化相機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 8000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 6000)
cnt = 0
stop_flag = False  # 全局停止旗標
model = None

def load_model(model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

# 建立目錄，如果不存在
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 根據預測類別保存圖片到對應資料夾
def save_image(image, predicted_class, timestamp, predictions):
    base_dir = os.path.join(os.getcwd(), str(predicted_class))

    if np.max(predictions) < 0.9:
        base_dir = os.path.join(os.getcwd(), "confusion")
        base_dir = os.path.join(base_dir, str(predicted_class))

    create_dir(base_dir)  # 確保目錄存在
    filename = os.path.join(base_dir, f"photo_{timestamp}.jpg")

    cv2.imwrite(filename, image)
    print(f'圖片已保存到: {filename}')

# 預測圖片類別
def predict(img, tag=''):
    model_path = 'cnn_200.h5'
    # model = tf.keras.models.load_model(model_path)
    load_model(model_path)

    # 圖片格式轉換
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 標準化處理

    # 預測類別
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predictions = predictions[0]
    predicted_class = predicted_class[0] + 1
    print(f'{tag} Predicted probability: {predictions}')
    print(f'{tag} Predicted class: {predicted_class}')
    return predicted_class, predictions

# 拍攝照片並保存
def take_picture():
    global cnt
    if not cap.isOpened() or stop_flag:
        print("停止拍攝或無法抓取鏡頭")
        return

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 進行預處理和預測
        r1, r2 = img_pre_proc.pre_proc(frame, timestamp)
        predicted_class, predictions = predict(r2, 'ROI 2')
        # 保存圖片到對應資料夾
        save_image(frame, predicted_class, timestamp, predictions)
        cnt += 1
        print(f"已拍攝照片並保存 [{cnt}]")
    else:
        print("無法讀取鏡頭 frame")

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_path)
    while(True):
        time.sleep(2)
        take_picture()
