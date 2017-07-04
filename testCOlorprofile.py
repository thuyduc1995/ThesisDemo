#!/usr/bin/python
# -*- coding: utf8-*-

import cv2
import numpy as np


s = '™'
s=unicode(s, 'utf-8')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,oriFrame=cap.read()
    frame=cv2.flip(oriFrame,1)

    frame=cv2.resize(frame,(960,720),interpolation = cv2.INTER_CUBIC)

    #cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 0), -1)
    cv2.putText(frame, '0', (550, 500), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '.', (630, 500), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '<-',(680, 495), font, 0.8  ,   (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '=', (770, 500), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '1', (550, 430), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '2', (620, 430), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '3', (690, 430), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '+', (770, 430), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '4', (550, 360), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '5', (620, 360), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '6', (690, 360), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '-', (770, 360), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '7', (550, 290), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '8', (620, 290), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '9', (690, 290), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '/', (780, 290), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '%', (550, 220), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,s.decode(), (620, 220), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'S', (690, 220), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, '1/x', (780, 220), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('masfk',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
