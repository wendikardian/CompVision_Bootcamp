import cv2
import HandDetector as hd
handDetect = hd.handDetector(detection_confident=0.8)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = handDetect.findHands(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()