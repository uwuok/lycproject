import tensorflow as tf

# 設定資料集路徑
dataset_path = r"C:\Users\user\Desktop\cnn\dataset_full"  # 替換為你的圖片資料集路徑

# 讀取圖片資料集，保持原始大小
batch_size = 16
img_height = 348  # 根據需要調整圖片高度
img_width = 149   # 根據需要調整圖片寬度
num_classes = 5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),  # 可以保持原始大小，不需改變
    batch_size=batch_size,
    color_mode='grayscale'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),  # 同樣保持原始大小
    batch_size=batch_size,
    color_mode='grayscale'
)

# 正規化圖片
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 根據需要調整模型結構
model = tf.keras.models.Sequential([
    # 使用卷積層來處理圖像數據
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),  # 使用具體的輸入形狀
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), # 攤平層
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)  # 將輸出層設為5個類別
])


#

# 編譯模型
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 訓練模型
history = model.fit(train_ds, validation_data=val_ds, epochs=121)


# 進行預測
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# 獲取一個批次的樣本
for images, labels in val_ds.take(1):
    predictions = probability_model(images)  # 使用模型進行預測
    # 輸出預測結果
    print(predictions.numpy())  # 將預測結果轉換為 numpy 陣列以便查看
    break  # 確保只執行一次迴圈



# 儲存模型
model.save('my_model.h5')


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager
# import wget 
# wget.download('https://github.com/GrandmaCan/ML/raw/main/Resgression/ChineseFont.ttf')

fontManager.addfont('ChineseFont.ttf')
mpl.rc('font', family='ChineseFont')

# 繪製損失圖
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.title('損失隨著訓練輪次的變化')
plt.xlabel('訓練輪次')
plt.ylabel('損失')
plt.legend()

# 繪製準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.title('準確率隨著訓練輪次的變化')
plt.xlabel('訓練輪次')
plt.ylabel('準確率')
plt.legend()

# 顯示圖表
plt.show()
