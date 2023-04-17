import cv2
import time
import HandDetection as hd
handDetect = hd.handDetector(detection_confident=0.8)

# cap = cv2.VideoCapture("video3.mp4")
cap = cv2.VideoCapture(0)
top_idx = [4,8,12,16,20]
previous_time = 0
current_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    frame = handDetect.findHands(frame)
    lmlist = handDetect.getHandLocation(frame, draw=True)
    # print(lmlist)
    if len(lmlist) != 0:
        fingers = []
        if lmlist[top_idx[0]][1] < lmlist[top_idx[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for idx in range(1,5):
            if lmlist[top_idx[idx]][2] < lmlist[top_idx[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, "frame rate: "+str(int(fps)), (350,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

