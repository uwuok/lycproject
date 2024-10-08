import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from collections import defaultdict
import img_pre_proc

# model = tf.keras.models.load_model('my_model.h5')
# print(model.summary())

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

model_path = r'my_model.h5'
model = tf.keras.models.load_model(model_path)

# key: 量級, value = 檔名
err = defaultdict(list)

def multi_file():
    cnt = 0
    files = os.listdir(current_path)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    for file in files:
        if file.lower().endswith(image_extensions):
            cnt += 1
            print(f'number of precessed image:{cnt}')
            print(f'name of current image:{file}')

            # current_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            r1, r2 = img_pre_proc.pre_proc(file)
            img_height = 348
            img_width = 149
            # 加載並處理圖片
            img = image.load_img(r1, target_size=(img_height, img_width), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # 標準化處理 [0, 1]
            predictions = model.predict(img_array)  # logits (邏輯回歸)
            predicted_class = np.argmax(predictions, axis=1)

            print(predictions)
            print(f'Predicted class:{predicted_class}')
            if predictions[0] != 0:
                err[predictions[0]].append(file)
    log = open('image/CNN/logfile.txt', mode='a')
    log.writelines(err)
    log.close()


def single_file():
    # img_path = r'dataset_full\1\photo_20240628_103404_1.jpg' # class 0
    # img_path = r'dataset_full\2\photo_20240628_111217_1.jpg' # class 1
    # img_path = r'dataset_full\3\photo_20240718_162934_1.jpg' # class 2
    # img_path = r'dataset_full\4\photo_20240718_154010_1.jpg' # class 3
    img_path = r'dataset_full\5\photo_20240628_113213_1.jpg'  # class 4


    img_height = 348
    img_width = 149
    # 加載並處理圖片
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 標準化處理 [0, 1]

    predictions = model.predict(img_array)  # logits (邏輯回歸)
    predicted_class = np.argmax(predictions, axis=1)

    print(predictions)
    print(f'Predicted class:{predicted_class}')


single_file()
