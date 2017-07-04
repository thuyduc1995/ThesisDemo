import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('E:\Setup File\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
count=0
list=[]
def takecenter(gray_frame,frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ox, oy, ow, oh) in faces:
        ox = ox + int(ow * 0.2)
        ow = int(ow * 0.6)
        oy = oy + int(oh * 0.2)
        oh = int(oh * 0.6)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
        cx=ox+int(ow/2)
        cy = oy + int(oh / 2)
        cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
        return cx,cy

def solve(cx,cy):
    global count


ret, frame = cap.read()
frame=cv2.flip(frame,1)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cx1,cy1=takecenter(gray_frame,frame) # 1st frame
list.append((cx1,cy1))
while (True):
    count=count+1
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ox, oy, ow, oh) in faces:
        ox = ox + int(ow * 0.2)
        ow = int(ow * 0.6)
        oy = oy + int(oh * 0.2)
        oh = int(oh * 0.6)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
        cx = ox + int(ow / 2)
        cy = oy + int(oh / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    if count==20:
        list.append((cx,cy))
        if list[-1][0]-list[-2][0]<-12 and list[-1][0]-list[0][0]<-12:
            print 'left'
        elif list[-1][0]-list[-2][0]>12 and list[-1][0]-list[0][0]>12:
            print 'right'
        else:
            print 'Normal'
        count=0

    cv2.imshow('new',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      # Display the resulting frame
cap.release()
cv2.destroyAllWindows()