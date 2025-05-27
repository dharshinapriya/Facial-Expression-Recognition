import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Load the dataset
df = pd.read_csv(r'C:\Users\lucky\facial_expression_recognition\fer2013\data\fer2013.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Convert pixel values from string to arrays and normalize them
def convert_pixels_to_array(pixels):
    return np.array(pixels.split(), dtype='float32').reshape(48, 48)

# Apply the conversion function to the 'pixels' column
df['pixels'] = df['pixels'].apply(convert_pixels_to_array)

# Normalize pixel values to [0, 1]
df['pixels'] = df['pixels'].apply(lambda x: x / 255.0)

# Split the dataset into training and testing data
X = np.array(df['pixels'].tolist())
y = df['emotion'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Reshape data for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)  # Adding channel dimension
X_test = X_test.reshape(-1, 48, 48, 1)

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 7 emotions

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.2f}")
