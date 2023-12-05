import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Food-101 dataset
data_path = '/path/to/food-101'
meta_path = '/path/to/food-101/meta'

df = pd.read_csv(meta_path + '/food-101/meta/train.txt', delimiter=' ', header=None)
df.columns = ['Image', 'Label']

images = []
labels = []

for index, row in df.iterrows():
    image_path = data_path + '/food-101/images/' + row['Image'] + '.jpg'
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    images.append(image)
    labels.append(row['Label'])

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model for food recognition
model_food_recognition = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(101, activation='softmax')  # Assuming 101 classes in Food-101
])

# Compile the model
model_food_recognition.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_food_recognition = model_food_recognition.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Define the regression model for calorie estimation
model_calorie_estimation = models.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model_calorie_estimation.compile(optimizer='adam', loss='mean_squared_error')

# Normalize labels for regression
scaler = StandardScaler()
y_train_normalized = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_normalized = scaler.transform(y_test.reshape(-1, 1))

# Train the model
history_calorie_estimation = model_calorie_estimation.fit(X_train, y_train_normalized, epochs=10, validation_data=(X_test, y_test_normalized))

# Evaluate the food recognition model
test_loss, test_acc = model_food_recognition.evaluate(X_test, y_test)
print(f'Food Recognition Test accuracy: {test_acc}')

# Evaluate the calorie estimation model
test_loss_calorie, _ = model_calorie_estimation.evaluate(X_test, y_test_normalized)
print(f'Calorie Estimation Test loss: {test_loss_calorie}')

# Plot training history for food recognition
plt.plot(history_food_recognition.history['accuracy'], label='accuracy')
plt.plot(history_food_recognition.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training history for calorie estimation
plt.plot(history_calorie_estimation.history['loss'], label='loss')
plt.plot(history_calorie_estimation.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
