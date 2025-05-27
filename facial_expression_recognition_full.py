import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('C:\\Users\\lucky\\facial_expression_recognition\\fer2013\\data\\fer2013.csv')

# Normalize pixel values to [0, 1]
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32') / 255.0)

# Prepare features and labels
X = np.array(df['pixels'].tolist())
y = np.array(df['emotion'].tolist())
X = X.reshape(-1, 48, 48, 1)  # Reshape for CNN

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # Assuming 7 emotion classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Save the model
model.save('facial_expression_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Example of making a prediction (uncomment and update the path below to test with a specific image)
# image_path = 'path_to_image'  # Replace with actual path
# new_image = preprocess_image(image_path)
# predictions = model.predict(new_image)
# predicted_class = np.argmax(predictions)
# print(f'Predicted emotion class: {predicted_class}')

# Visualize training metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
