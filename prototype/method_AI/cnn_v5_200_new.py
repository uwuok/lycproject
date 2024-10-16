import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager

# 設定資料集路徑
train_dataset_path = r'C:\Users\user\Desktop\cnn\切好的影像\dataset_200\_training_data'
validation_data_path = r'C:\Users\user\Desktop\cnn\切好的影像\dataset_200\_validation_data'

# 讀取圖片資料集，保持原始大小
batch_size = 8
img_height = 348  # 根據需要調整圖片高度
img_width = 149  # 根據需要調整圖片寬度
num_classes = 5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_path,
    seed=12173,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_path,
    seed=12173,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)

# 正規化圖片
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 根據需要調整模型結構
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation='relu', input_shape=(img_height, img_width, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# 編譯模型
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


# 自定義回調以計算每個類別的準確率
class ClassAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes):
        super(ClassAccuracyCallback, self).__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.class_accuracy = {i: [] for i in range(num_classes)}

    def on_epoch_end(self, epoch, logs=None):
        total_correct = np.zeros(self.num_classes)
        total_samples = np.zeros(self.num_classes)

        for images, labels in self.val_ds:
            predictions = np.argmax(self.model.predict(images), axis=1)
            for i in range(len(labels)):
                total_samples[labels[i]] += 1
                if predictions[i] == labels[i]:
                    total_correct[labels[i]] += 1

        for i in range(self.num_classes):
            if total_samples[i] > 0:
                self.class_accuracy[i].append(total_correct[i] / total_samples[i])
            else:
                self.class_accuracy[i].append(0)


# 實例化回調
class_accuracy_callback = ClassAccuracyCallback(val_ds, num_classes)

# 訓練模型
history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[class_accuracy_callback])

fontManager.addfont('ChineseFont.ttf')
mpl.rc('font', family='ChineseFont')

# 繪製損失圖
plt.figure()
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.title('損失隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('損失')
plt.xticks(np.arange(0, len(history.history['loss']), step=10))
plt.legend()
plt.show()

# 繪製整體準確率圖
plt.figure()
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.title('準確率隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('準確率')
plt.xticks(np.arange(0, len(history.history['accuracy']), step=10))
plt.legend()
plt.show()

# 繪製各類別準確率圖
plt.figure()
for i in range(num_classes):
    plt.plot(class_accuracy_callback.class_accuracy[i], label=f'類別 {i} 準確率')
plt.title('各類別準確率隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('準確率')
plt.xticks(np.arange(0, len(class_accuracy_callback.class_accuracy[0]), step=10))
plt.legend()
plt.show()

# 進行預測
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# 獲取一個批次的樣本
for images, labels in val_ds.take(1):
    predictions = probability_model(images)  # 使用模型進行預測
    break  # 確保只執行一次迴圈

# 輸出預測結果
print(predictions.numpy())  # 將預測結果轉換為 numpy 陣列以便查看

# 儲存模型
model.save('cnn_200.h5')
