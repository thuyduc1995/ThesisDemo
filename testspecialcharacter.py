import cv2
import numpy as np


type=''
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
calc=cv2.imread('calof1.jpg')
while True:
    ret,oriFrame=cap.read()
    frame=cv2.flip(oriFrame,1)
    frame=cv2.resize(frame,(960,720),interpolation = cv2.INTER_CUBIC)
    if type == 'c':
        cv2.rectangle(frame, (600, 516), (681, 572), (0,0, 0),-1)
    if type == 'b':
        cv2.rectangle(frame, (681, 516), (762, 572), (0,0,0), -1)
    if type == 'v':
        cv2.rectangle(frame, (600, 516), (681, 572), (0, 0, 0), -1)
        cv2.rectangle(frame, (681, 516), (762, 572), (0, 0, 0), -1)
    duc=frame[50:570,600:910]
    dst = cv2.addWeighted(calc, 0.8, duc, 0.2, 0)
    frame[50:570, 600:910]=dst

    cv2.imshow('masfk',frame)

    k = cv2.waitKey(5) & 0xFF

    if k == ord('c'):
        type='c'
    if k == ord('b'):
        type='b'
    if k == ord('v'):
        type='v'



    elif k == 27:
        break

cv2.destroyAllWindows()
cap.release()
