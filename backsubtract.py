import numpy as np
import cv2
cap = cv2.VideoCapture(0)


while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = (31, 31)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresholded = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('frame',thresholded)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()