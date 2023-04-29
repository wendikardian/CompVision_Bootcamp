import silence_tensorflow.auto
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_keras
from collections import deque
import mediapipe as mp
import numpy as np
import cv2


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
Q = deque(maxlen=10)
writer = None

# parameters for loading data and images
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral")
emotion_model_path = 'models/_trained.hdf5'
out_video_path = 'output/video.avi'

# loading models
model = load_model(emotion_model_path, compile=False)


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("vid1.mp4")


with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        check, frame = cap.read()

        # if not check:
        #     print("Ignoring empty camera frame.")
        #     # If loading a video, use 'break' instead of 'continue'.
        #     continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
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
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy

                # crop detected face
                detected_face = frame[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
                detected_face = cv2.cvtColor(
                    detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
                detected_face = cv2.resize(detected_face, (64, 64))

                frame_pixels = img_keras.img_to_array(detected_face)
                frame_pixels = np.expand_dims(frame_pixels, axis=0)

                # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                frame_pixels /= 255

                # store probabilities of 7 expressions
                emotion = model.predict(frame_pixels)[0]
                Q.append(emotion)

                # perform prediction averaging over the current history of previous predictions
                results = np.array(Q).mean(axis=0)
                i = np.argmax(results)
                label = emotions[i]

                # write emotion text above rectangle
                cv2.putText(frame, label, (cx_min, cy_min),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)
                #cv2.circle(image, ((cx_min+cx_max)//2, (cy_min+cy_max)//2), 100, (0, 255, 0), 2)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            h, w, c = frame.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            writer = cv2.VideoWriter(out_video_path, fourcc, 20, (w, h), True)

        # write the output frame to disk
        writer.write(frame)

        cv2.imshow('MediaPipe FaceMesh', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
