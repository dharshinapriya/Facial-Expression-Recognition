# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the CSV file
data_path = r"C:\Users\lucky\facial_expression_recognition\fer2013\data\fer2013.csv"

# Step 1: Load the dataset
try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")

# Step 2: Inspect the data
print("First few rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nUnique emotion labels:", df['emotion'].unique())

# Step 3: Preprocess the Data

# Convert the pixels from a string to a numpy array
def preprocess_pixels(pixels):
    # Convert space-separated pixel values to a numpy array of floats
    pixels_array = np.array(pixels.split(), dtype='float32')
    # Reshape the array to 48x48 (assuming 48x48 image dimensions)
    pixels_array = pixels_array.reshape(48, 48)
    # Normalize the pixel values (0 to 1 range)
    pixels_array = pixels_array / 255.0
    return pixels_array

# Apply preprocessing to all rows in the dataset
df['pixels'] = df['pixels'].apply(preprocess_pixels)

# Convert emotion labels to categorical (if necessary)
# You can use a dictionary to map emotion labels to names if needed.
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
df['emotion_label'] = df['emotion'].map(emotion_labels)

# Display some sample images with labels to verify preprocessing
print("\nSample images after preprocessing:")
for i in range(5):
    plt.imshow(df['pixels'].iloc[i], cmap='gray')
    plt.title(f"Emotion: {emotion_labels[df['emotion'].iloc[i]]}")
    plt.show()
