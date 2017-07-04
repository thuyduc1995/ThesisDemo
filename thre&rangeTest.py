import cv2
import numpy as np

cap=cv2.VideoCapture(0)


while True:
    ret,duc=cap.read()
    grey = cv2.cvtColor(duc, cv2.COLOR_BGR2GRAY)
    value = (31, 31)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresholded = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    cv2.imshow('duc',duc)
    cv2.imshow('duc1', grey)
    cv2.imshow('duc2', thresholded)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()