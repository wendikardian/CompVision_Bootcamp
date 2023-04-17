import cv2
import mediapipe as mp
class handDetector:
    def __init__(self, static_mode=False, maxhands=2,
    detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.maxhands = maxhands
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.static_mode,
        self.maxhands, self.detection_confident, self.
        tracking_confident)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw_landmark=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, handlms,
                    self.mphands.HAND_CONNECTIONS)
        return frame

