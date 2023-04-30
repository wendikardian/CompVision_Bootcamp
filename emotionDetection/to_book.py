import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture("vid1.mp4")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        check, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame)
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
                print(face_landmarks.landmark)
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
                detected_face = frame[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
                detected_face = cv2.cvtColor(
                    detected_face, cv2.COLOR_BGR2GRAY)
                detected_face = cv2.resize(detected_face, (64, 64))
        cv2.imshow('frame', frame)
        cv2.imshow('Detected face', detected_face)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()