# Import necessary libraries
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import cvzone
from cvzone.FaceDetectionModule import FaceDetector



# Load the trained model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load trained weights
model.load_weights('model.h5')

# Load face cascade for face detection
#facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = FaceDetector()



# Define emotion categories
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
stress_emotions = {"Angry", "Disgusted", "Fearful", "Sad"}  # Emotions that contribute to stress

# Initialize stress tracking
total_frames = 0
stress_count = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img, bboxs = detector.findFaces(frame)
    faces = []
    for i in range(len(bboxs)):
        faces.append(bboxs[i]["bbox"])

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]

        # Draw emotion text
        #cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x + 75, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Update stress tracking
        total_frames += 1
        if emotion in stress_emotions:
            stress_count += 1

    # Calculate stress percentage
    stress_percentage = (stress_count / total_frames) * 100 if total_frames > 0 else 0

    # Display stress percentage as text
    cv2.putText(frame, f"Stress Level: {stress_percentage:.2f}%", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Create a simple graphical representation (progress bar) for stress
    bar_x, bar_y, bar_w, bar_h = 20, 100, 300, 30  # Set position and size of bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)  # Outline of bar
    fill_w = int(bar_w * (stress_percentage / 100))  # Calculate filled portion based on stress percentage
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)  # Fill bar with color

    # Show the frame
    cv2.imshow('Emotion & Stress Detector', frame)

    

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
