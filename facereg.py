import cv2
import numpy as np

cap = cv2.VideoCapture(0)
i=1
face_cascade = cv2.CascadeClassifier('E:\Setup File\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    duc=frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Convert to gray
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    x,y,w,h=faces[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(duc, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow('frame',frame)
    cv2.imshow('frame1', duc)
    if cv2.waitKey(1) & 0xFF ==27:
        break
cap.release()
cv2.destroyAllWindows()