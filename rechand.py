import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    img=cv2.imread('l.png')
    # Take each frame
    _, frame = cap.read()
    duc=frame[300:400,400:500]

    img[200:300,200:300]=duc
    frame[100:400,200:500]=img


    cv2.imshow('frame',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()