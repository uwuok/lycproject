import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'找到{len(gpus)}個 GPU:{gpus}')
else:
    print('cant find any gpu')