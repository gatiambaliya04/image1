# Assume this is your predict.py file content
# Save this content in a file named "predict.py"

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("image_classification_model.h5")

# Load and preprocess a sample image (you can replace this with your own image)
sample_image_path = "path/to/your/sample/image.jpg"
sample_image = plt.imread(sample_image_path)
sample_image = tf.image.resize(sample_image, (32, 32))
sample_image = sample_image / 255.0
sample_image = tf.expand_dims(sample_image, 0)  # Add batch dimension

# Make predictions
predictions = model.predict(sample_image)

# Display the prediction
predicted_class_index = tf.argmax(predictions[0])
predicted_class = class_names[predicted_class_index]
confidence = tf.reduce_max(tf.nn.softmax(predictions[0])) * 100

plt.imshow(sample_image[0])
plt.title(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.show()
