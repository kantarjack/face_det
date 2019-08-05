import cv2
import face_recognition
import pyttsx3
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) 
path='trainingData'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.24-0.77.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["Sad","Happy","Neutral"]
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
names=[]
known_faces=list()
for imagePath in imagePaths:
        name_temp=str(os.path.split(imagePath)[-1].split(".")[0])
        names.append(name_temp)
        image = cv2.imread(imagePath)
        known_faces.append(face_recognition.face_encodings(image)[0])

face_locations = []
face_encodings = []
face_names = []

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

while True:
    ret, frame =cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for (top, right, bottom, left),face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        name = "Unknown"
        if(True in match):
            name=names[match.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, name, (left + 6, bottom - 30), font, 0.5, (255, 255, 255), 1)
       
        if (name!='Unknown'):
            engine.say('Hello '+name+'... Welcome!!!')
            engine.runAndWait()
    cv2.imshow('camera',frame)
    k = cv2.waitKey(10) & 0xff
    if (k == 27 or k==ord('q')):
        break
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()