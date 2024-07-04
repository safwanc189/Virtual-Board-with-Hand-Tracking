import cv2
import HandTrackingModule as htm
import numpy as np

detector = htm.HandDetector()

drawColor = (0,0,255)
brushSize = 5
eraserSize = 50
imgCanvas = np.zeros((720,1280,3),np.uint8)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(1280,720))
    frame = cv2.rectangle(frame,(0,0),(1280,110),(0,0,0),-1)
    frame = cv2.rectangle(frame,(20,10),(210,100),(0,0,255),-1)
    frame = cv2.rectangle(frame,(230,10),(450,100),(0,255,0),-1)
    frame = cv2.rectangle(frame,(470,10),(680,100),(255,0,0),-1)
    frame = cv2.rectangle(frame,(700,10),(920,100),(0,255,255),-1)
    frame = cv2.rectangle(frame,(940,10),(1270,100),(255,255,255),-1)
    frame = cv2.putText(frame,'ERASER',(1050,65),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)

    # 1. find hands, landmarks
    frame = detector.find_hands(frame)
    lmlist = detector.find_position(frame)

    if len(lmlist)!=0:
        # tip of index and middle finger
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

    # 2. check if finger is up
    fingers = detector.fingers_up(lmlist)

    # 3. if 2 fingers are up ==> selection mode
    if fingers[1] and fingers[2]:
        print('selection mode')

        xp, yp = 0, 0

        if y1 < 120:
            if 20 < x1 < 210:
                print('red')
                drawColor = (0,0,255)
            elif 230 < x1 < 450:
                print('green')
                drawColor = (0,255,0)
            elif 470 < x1 < 680:
                print('blue')
                drawColor = (255,0,0)
            elif 700 < x1 < 920:
                print('yellow')
                drawColor = (0,255,255)
            else:
                print('eraser')
                drawColor = (0,0,0)

        cv2.rectangle(frame,(x1,y1),(x2,y2),drawColor,-1)

    # 4. if one finger is up ==> drawing mode
    if fingers[1] and not fingers[2]:
        print('Drawing mode')

        cv2.circle(frame,(x1,y1),15,drawColor,-1)
        if xp == 0 and yp == 0:
            xp = x1
            yp = y1

        if (drawColor == (0,0,0)):
            cv2.line(frame,(xp,yp),(x1,y1),drawColor,eraserSize)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserSize)
        else:
            cv2.line(frame,(xp,yp),(x1,y1),drawColor,brushSize)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushSize)

        xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,imgInv)
    frame = cv2.bitwise_or(frame,imgCanvas)

    cv2.imshow('virtual board',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
