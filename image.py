import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 3: Load and Explore the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training set shape:", train_images.shape)
print("Testing set shape:", test_images.shape)

plt.figure()
plt.imshow(train_images[0], cmap='gray')
plt.title(class_names[train_labels[0]])
plt.show()

# Step 4: Preprocess the Data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 5: Build the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Step 8: Make Predictions
predictions = model.predict(test_images)

for i in range(5):
    plt.figure(figsize=(2, 2))
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"True: {class_names[test_labels[i]]}\nPredicted: {class_names[tf.argmax(predictions[i])]}")
    plt.show()
