import mediapipe as mp
import numpy as np
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_keras
from collections import deque

# cap = cv2.VideoCapture("vid1.mp4")
cap = cv2.VideoCapture(0)
writer = None
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model = load_model('models/_trained.hdf5', compile=False)
Q = deque(maxlen=10)
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad",
"Surprise", "Neutral")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1,
circle_radius=0)
with mp_face_mesh.FaceMesh(
min_detection_confidence=0.5,
min_tracking_confidence=0.5) as face_mesh:
    while True:
        check, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
                h, w, c = frame.shape
                cx_min=w
                cy_min=h
                cx_max=cy_max=0
                # print(cx_min, cy_min, cx_max, cy_max)
                for id, lm in enumerate(face_landmarks.landmark):
                    # print(lm.x, lm.y)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                # print(cx_min, cy_min, cx_max, cy_max)
                detected_face = frame[int(cy_min):int(cy_max), int(
                cx_min):int(cx_max)]
                detected_face = cv2.cvtColor(detected_face,
                cv2.COLOR_BGR2GRAY)
                detected_face = cv2.resize(detected_face, (64, 64))
                frame_pixels = img_keras.img_to_array(detected_face)
                frame_pixels = np.expand_dims(frame_pixels, axis=0)
                frame_pixels /= 255
                emotion = model.predict(frame_pixels)[0]
                Q.append(emotion)
                # print(Q)
                results = np.array(Q).mean(axis=0)
                i = np.argmax(results)
                label = emotions[i]
                # print(label)
                cv2.putText(frame, label, (cx_min, cy_min),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max),
                (0, 255, 0), 2)
        if writer is None:
            h, w, c = frame.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            writer = cv2.VideoWriter('output.avi', fourcc, 20,
            (w, h), True)
        writer.write(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('Detected face', detected_face)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()