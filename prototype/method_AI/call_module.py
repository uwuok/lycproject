import cv2
import tensorflow as tf
import numpy as np
import os
from collections import defaultdict
import img_pre_proc
import matplotlib.pyplot as plt

# model = tf.keras.models.load_model('my_model.h5')
# print(model.summary())

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

model_path = 'cnn_complete_rm50.h5'

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for model at: {os.path.abspath(model_path)}")


model = tf.keras.models.load_model(model_path)

# key: 量級, value = 檔名
success_r1, success_r2, fail_r1, fail_r2 = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
err = defaultdict(list)


def multi_file():
    cnt = 0
    files = os.listdir(current_path)
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
    for file in files:
        if file.lower().endswith(image_extensions):
            cnt += 1
            print(f'Number of processed images: {cnt}')
            print(f'Name of current image: {file}')

            current_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            r1, r2 = img_pre_proc.pre_proc(current_img)
            img_height = 348
            img_width = 149

            # 加載並處理圖片
            img = r1
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # 標準化處理 [0, 1]

            # 預測類別
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)

            print(predictions)
            print(f'Predicted class: {predicted_class + 1}')

            # 使用預測結果
            if predicted_class[0] + 1 == 1:
                success_r1[predicted_class[0] + 1] += 1
            else:
                err[predicted_class[0] + 1].append(file)
                fail_r1[predicted_class[0] + 1] += 1

    plot_white_ratio_histogram()

    # 寫入日誌
    with open('logfile.txt', mode='a') as log:
        for key, value in err.items():
            log.write(f"Class {key}: {', '.join(value)}\n")
    

def plot_white_ratio_histogram():
    global success_r1, success_r2, fail_r1, fail_r2
    white_ratios_r1 = []
    white_ratios_r2 = []
    
    # 將 success_r1 和 fail_r1 中的 white_ratio 值加入列表
    for k in success_r1:
        white_ratios_r1.extend([k] * success_r1[k])
    for k in fail_r1:
        white_ratios_r1.extend([k] * fail_r1[k])
    
    # 調整圖形的大小，讓它看起來更平衡
    # plt.figure(figsize=(8, 5))  # 調整為 8x5 的圖像大小
    
    # 繪製直方圖，使用 rwidth 調整柱子之間的距離
    plt.hist(white_ratios_r1, bins=range(1, 7), edgecolor='black', alpha=0.7, label='r1', color='blue', rwidth=1)
    
    # 添加圖表標題和軸標籤
    plt.title('Histogram of weight class for r1')
    plt.xlabel('Weight class')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 7, 1), horizontalalignment='center')  # 確保標籤水平對齊於中間
    
    # 添加圖例來區分 r1 和 r2
    plt.legend()
    
    # 添加網格線
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 調整 x 軸標籤的位置
    plt.gca().tick_params(axis='x', pad=10)  # `pad=10` 增加標籤與 x 軸之間的距離
    
    # 顯示圖表
    plt.show()


def single_file(img):
    # img_path = r'dataset_full\1\photo_20240628_103404_1.jpg' # class 0
    # img_path = r'dataset_full\2\photo_20240628_111217_1.jpg' # class 1
    # img_path = r'dataset_full\3\photo_20240718_162934_1.jpg' # class 2
    # img_path = r'dataset_full\4\photo_20240718_154010_1.jpg' # class 3
    # img_path = r'dataset_full\5\photo_20240628_113213_1.jpg' # class 4

    img_height = 348
    img_width = 149
    # 加載並處理圖片
    # img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    # img_array = tf.keras.utils.img_to_array(img_path)
    img_array = cv2.resize(img, (img_width, img_height))  # 調整大小
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 標準化處理 [0, 1]

    predictions = model.predict(img_array)  # logits (邏輯回歸)
    predicted_class = np.argmax(predictions, axis=1)

    print(predictions)
    print(f'Predicted class:{predicted_class[0] + 1}')
    return predicted_class[0] + 1

if __name__ == '__main__':
    multi_file()
