import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fake_detection_model.h5')

# Function to preprocess an image for inference
def preprocess_image(image_path, target_size=(32, 32)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the pixel values to [0, 1]
    return img

# Function to classify an image as AI-generated (0) or real (1)
def classify_image(image_path):
    image = preprocess_image(image_path)
    
    # Make predictions
    prediction = model.predict(image)

    return prediction[0][0]  # The value on a scale from 0 to 1

# Example usage
# image_path = 'D:\\Work and Stuff\\Works\\Image Realism\\test1.jpg'
image_path = "E:\\Huraira\\Advanced Algorithms\\Image Realism\\testreal.jpg"

rating = classify_image(image_path)
print(f'The image is rated as {rating:.2f}, where 0 means AI-generated, and 1 means completely real.')
