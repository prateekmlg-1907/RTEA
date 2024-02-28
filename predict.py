# # Import necessary libraries
# import cv2
# from keras.models import load_model
# import numpy as np
#
# # Load the model
# model = load_model('models/emotion_detector.h5')
#
# # Define the list of labels
# labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# # Load the image
# img = cv2.imread("E:\Downloads\IMG_1941 (1).jpg")
#
# # Convert color to gray
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Detect faces
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
# for (x, y, w, h) in faces:
#     # Get face image
#     face_img = gray[y:y+h, x:x+w]
#     face_img = cv2.resize(face_img, (48, 48))
#     face_img = face_img.astype('float') / 255.0
#     face_img = np.asarray(face_img)
#     face_img = face_img.reshape(1, 1, face_img.shape[0], face_img.shape[1], face_img.shape[2])
#     face_img = np.vstack([face_img])
#
#     # Predict emotion
#     prediction = model.predict(face_img)[0]
#     label = labels[prediction.argmax()]
#
#     # Print the prediction
#     print("Predicted emotion: ", label)
