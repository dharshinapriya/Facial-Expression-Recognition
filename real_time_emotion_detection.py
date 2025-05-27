import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('C:/Users/lucky/facial_expression_recognition/model.h5')  # Ensure to update the path if necessary

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (face) for emotion detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match the model input
        roi_gray = roi_gray.astype('float32') / 255.0  # Normalize the image
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Expand dimensions to match model input shape
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension

        # Predict emotion
        emotion_prediction = model.predict(roi_gray)
        max_index = np.argmax(emotion_prediction[0])
        emotion_label = emotion_labels[max_index]

        # Display the predicted emotion label
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
