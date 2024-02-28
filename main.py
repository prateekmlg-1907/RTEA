# Import necessary libraries
import cv2
from keras.models import load_model
import numpy as np

# Load the model
model = load_model('models/emotion_detector.h5')

# Define the list of labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert color to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Get face image
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float') / 255.0
        face_img = np.asarray(face_img)
        face_img = face_img.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(face_img)[0]
        label = labels[prediction.argmax()]

        # Draw rectangle around the face and put label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
