import cv2
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
tracked = False
checkframe_pos = np.array([[600, 120], [600, 260], [650, 150], [620, 320], [570, 210], [640, 230], [660, 250]])
nocheckframe=checkframe_pos.__len__()
font = cv2.FONT_HERSHEY_SIMPLEX
calc=cv2.imread('testcal2.jpg')
kernel = np.ones((5,5),np.uint8)
face_cascade = cv2.CascadeClassifier('E:\Setup File\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cflag_button=''
pflag_button=''
start_time=time.time()
list_button=['0','1','2','3','4','5','6','7','8','9','+','-','x','/','%','sqrt','sqr','1/','ans','c','<-','+-','.','=','']
ps=''
cs='0'
caltrack=False
ans=''
resulted=False
strx=640
stry=0
moveCalpos=[0,0]
moveCal=False
duc = []
def drawCheckFrame(des, x, y, color, thickness):
    if color == 'r':
        colorcode = 0, 0, 255
    elif color == 'g':
        colorcode =0, 255, 0
    elif color == 'b':
        colorcode = 255, 0, 0
    else:
        colorcode =0,0,0
    cv2.rectangle(des, (x, y), (x+10, y+10), (colorcode), thickness)

def getProfColor(profile):
    h_value=[]
    s_value=[]
    v_value=[]

    for i in range(0,7):
        h_value.append(profile[i][0])
        s_value.append(profile[i][1])
        v_value.append(profile[i][2])
    h_value.sort()
    s_value.sort()
    v_value.sort()
    return h_value,s_value,v_value






def produceBoundarie(h,s,v):
    h_lower=10
    h_upper=10
    s_lower=35
    s_upper=35
    v_lower=60
    v_upper=60
    if h-h_lower<0:
        h_lower=h
    if s-s_lower<0:
        s_lower=s
    if v-v_lower<0:
        v_lower=v
    if h+h_upper>255:
        h_upper=255-h
    if s+s_upper>255:
        s_upper=255-s
    if v+v_upper>255:
        v_upper=255-v
    lower = np.array([h-h_lower, s-s_lower, v-v_lower])
    upper = np.array([h+h_upper, s+s_upper, v+v_upper])
    return lower,upper


while True:
    ret,oriFrame=cap.read()
    frame = cv2.flip(oriFrame,1)
    frame = cv2.resize(frame, (1060, 720), interpolation=cv2.INTER_CUBIC)
    if tracked is False:
        for i in range(0, nocheckframe):
            drawCheckFrame(frame, checkframe_pos[i][0], checkframe_pos[i][1], 'g', 1)
        cv2.imshow('Flipped',frame)

    else:
        drawframe = frame.copy()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(0,nocheckframe):
            lower,upper=produceBoundarie(profile_hand[i][0],profile_hand[i][1],profile_hand[i][2])
            mask = cv2.inRange(frame_hsv, lower, upper)
            duc.append(mask)
            if i==0:
                offmask = mask
            else:
                offmask=mask+offmask

        median = cv2.medianBlur(offmask, 29)
        #duc[0]=cv2.medianBlur(duc[0],5)
        #duc[1] = cv2.medianBlur(duc[1], 5)
        #duc[2] = cv2.medianBlur(duc[2], 5)
        #duc[3] = cv2.medianBlur(duc[3], 5)
        cv2.imshow('draw1', duc[0])
        cv2.imshow('draw2', duc[1])
        cv2.imshow('draw3', duc[2])
        cv2.imshow('draw4', duc[3])
        cv2.imshow('draw5', duc[4])
        cv2.imshow('draw6', duc[5])
        cv2.imshow('draw',drawframe)
    k = cv2.waitKey(3) & 0xFF


    if k==ord('c'):
        profile_hand = np.zeros(shape=(nocheckframe,3))
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        for i in range (0,nocheckframe):
            profile_hand[i]= np.array(frame_hsv[checkframe_pos[i][1]+5,checkframe_pos[i][0]+5])
        tracked = True
        cv2.destroyWindow('Flipped')
    elif k==ord('b'):
        profile_hand = np.zeros(shape=(nocheckframe, 3))
        tracked =False
    elif k == 27:
        break

cv2.destroyAllWindows()
cap.release()
