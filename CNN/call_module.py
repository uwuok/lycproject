import tensorflow as tf
import numpy as np

model_path = r'my_model.h5'
model = tf.keras.models.load_model(model_path)


def predict(img):
    # Load your model

    img_height = 348
    img_width = 149

    # Assuming 'img_array' is your NumPy array representing the image
    # Ensure img_array has the shape (height, width, channels)
    # For grayscale images, channels should be 1
    # If your image is already in the correct shape, you can proceed directly.

    # Example: Creating a dummy grayscale image (for demonstration)
    img_array = img

    # Expand dimensions to fit model input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data to [0, 1]
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)  # logits (logits)
    predicted_class = np.argmax(predictions, axis=1)

    print(predictions)
    print(f'Predicted class: {predicted_class}')
