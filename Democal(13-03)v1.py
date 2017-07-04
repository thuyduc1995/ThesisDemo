import cv2
import numpy as np
import math
import time
import win32api
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
strx=540
stry=0
moveCalpos=[0,0]
moveCal=False
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

def extractHandContour(contours):
    maxArea, index = 0, 0
    for i in xrange(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea = area
            index = i
    realHandContour = contours[index]
    realHandLen = cv2.arcLength(realHandContour, True)
    # reduce hand contour to manageable number of points
    # Thanks to http://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html
    handContour = cv2.approxPolyDP(realHandContour,
                                        0.001 * realHandLen, True)
    return handContour

def findHullAndDefects(im,handContour):
    hullHandContour = cv2.convexHull(handContour,returnPoints = False)
    hullPoints = [handContour[i[0]] for i in hullHandContour]
    hullPoints = np.array(hullPoints, dtype = np.int32)
    defects = cv2.convexityDefects(handContour,hullHandContour)
    #cv2.drawContours(im, [handContour], 0, (0, 255, 0), 3)
    cv2.drawContours(im, [hullPoints], 0, (0, 0, 255), 2)


def drawCenter(im,handContour):
    M=cv2.moments(handContour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(im, (cx, cy), 80, (0, 0, 255), 3)
    cv2.circle(im, (cx, cy), 4, (0, 255, 255), -1)
    cv2.putText(im, 'Center of hand', (cx+15, cy-15), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    return cx,cy

def distanceP2P(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def getAngle(x1,x2,y1,y2,z1,z2):
    l1=distanceP2P(y1,y2,x1,x2)
    l2=distanceP2P(y1,y2,z1,z2)
    dot=(x1-y1)*(z1-y1)+(x2-y2)*(z2-y2)
    angle=math.acos(dot/(l1*l2))
    angle=angle*180/math.pi
    return angle
def drawDefect(im,realHandContour):

    hull=cv2.convexHull(realHandContour,returnPoints= False)
    defects=cv2.convexityDefects(realHandContour,hull)
    x, y, w, h = cv2.boundingRect(realHandContour)
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    angleTol=140
    tolerance=h/6
    nodefects=0
    realDefect=defects.copy()
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(realHandContour[s][0])
        end   = tuple(realHandContour[e][0])
        far   = tuple(realHandContour[f][0])
        if ((distanceP2P(start[0],start[1],far[0],far[1])>tolerance) and (distanceP2P(end[0],end[1],far[0],far[1])>tolerance) and (getAngle(start[0],start[1],far[0],far[1],end[0],end[1]) < angleTol)):
            realDefect[nodefects,0]=defects[i,0]
            nodefects=nodefects+1

    for i in range (nodefects):
        for j in range(nodefects):
            s1, e1, f1, d1 = defects[i, 0]
            s2, e2, f2, d2 = defects[j, 0]
            start1 = tuple(realHandContour[s1][0])
            end1 = tuple(realHandContour[e1][0])
            start2 = tuple(realHandContour[s2][0])
            end2 = tuple(realHandContour[e2][0])
            if (distanceP2P(end1[0],end1[1],start2[0],start2[1])<w/6):
                realDefect[j,0,0]=realDefect[i,0,1]
                break
            if (distanceP2P(end2[0],end2[1],start1[0],start1[1])<w/6):
                realDefect[j, 0, 1] = realDefect[i, 0, 0]

    for i in range(nodefects):
        s, e, f, d = realDefect[i, 0]
        start = tuple(realHandContour[s][0])
        end = tuple(realHandContour[e][0])
        far = tuple(realHandContour[f][0])
        cv2.circle(im, start, 5, (255, 0, 0), 2)
        cv2.putText(im, '%d' % (i), (start), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(im, far, 5, (0, 255, 0), 2)
        cv2.putText(im, '%d' % (i), (far), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(im, end, 5, (0, 0, 255), 2)
        cv2.putText(im, '%d' % (i), (end), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # duc='Number of fingcer: %d' %(nodefects+1)
    # cv2.putText(im, duc, (20,20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return realDefect,nodefects

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
def drawbutton(im,btnno,x,y,size,tness):
    cv2.rectangle(im, (x, y), (x+size, y+size), (255, 0, 0), tness)
    cv2.putText(im, btnno, (x+15, y+35), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
def drawCalculator(im):
    size = 50
    y=200
    num=1
    for j in range (1,4):
        x = 400
        for i in range(1,4):
            drawbutton(im,'%d' %num,x,y,size,2)
            num=num+1
            x=x+size+10
        y=y-size-10
def clickCalculator(im,realHandContour,realDefect,nodefect):
    global start_time,cflag_button,pflag_button,caltrack, moveCal,moveCalpos,strx,stry
    if (nodefect>0):
        listEndPoint=[]
        for i in range (0,nodefect):
            listEndPoint.append(realDefect[i,0,1])
        listEndPoint.sort()
        end = tuple(realHandContour[listEndPoint[0]][0])
        cx =end[0]
        cy =end[1]
        #win32api.SetCursorPos((cx, cy))
        #win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        #win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        if (cx in range(strx,strx+320) and cy in range(stry,stry+ 126)):
            cv2.rectangle(im, (strx, stry), (strx+320, stry+126), (255, 0, 0), -1)
            if moveCal is False:
                moveCalpos[0]=cx
                moveCalpos[1]=cy
                moveCal=True
            else:
                strx= strx -(moveCalpos[0]-cx)
                stry= stry +(-moveCalpos[1]+cy)
                if stry<0:
                    stry=0
                if stry>282:
                    stry=282
                if strx>740:
                    strx=740
                if strx<0:
                    strx=0
                print strx,stry
                moveCal=False
        elif cy in range(stry+126,stry+ 178):
            if cx in range(strx,strx+80):
                cv2.rectangle(im, (strx, stry+126), (strx+80, stry+126+52), (255, 0, 0), -1)   #%
                cflag_button='%'
            elif cx in range(strx+80,strx+80+80):
                cv2.rectangle(im, (strx+80, stry+126), (strx+80+80, stry+126+52), (255, 0, 0), -1)   #sqrt
                cflag_button = 'sqrt'
            elif cx in range(strx+80+80,strx+80+80+80):
                cv2.rectangle(im, (strx+80+80, stry+126), (strx+80+80+80, stry+126+52), (255, 0, 0), -1)    #sqr
                cflag_button = 'sqr'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126), (strx+80+80+80+80, stry+126+52), (255, 0, 0), -1)    #1/
                cflag_button = '1/'
        elif cy in range(stry+126+52, stry+126+52+52):
            if cx in range(strx,strx+80) :
                cv2.rectangle(im, (strx, stry+126+52), (strx+80, stry+126+52+52), (255, 0, 0), -1)   #ans
                cflag_button = 'ans'
            elif cx in range(strx+80,strx+80+80) :
                cv2.rectangle(im, (strx+80, stry+126+52), (strx+80+80, stry+126+52+52), (255, 0, 0), -1)   #c
                cflag_button = 'c'
            elif cx in range(strx+80+80,strx+80+80+80) :
                cv2.rectangle(im, (strx+80+80, stry+126+52), (strx+80+80+80, stry+126+52+52), (255, 0, 0), -1)    #<-
                cflag_button = '<-'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126+52), (strx+80+80+80+80, stry+126+52+52), (255, 0, 0), -1)    #/
                cflag_button = '/'
        elif cy in range(stry+126+52+52, stry+126+52+52+52):
            if cx in range(strx,strx+80) :
                cv2.rectangle(im, (strx, stry+126+52+52), (strx+80, stry+126+52+52+52), (255, 0, 0), -1)   #7
                cflag_button = '7'
            elif cx in range(strx+80,strx+80+80) :
                cv2.rectangle(im, (strx+80, stry+126+52+52), (strx+80+80, stry+126+52+52+52), (255, 0, 0), -1)   #8
                cflag_button = '8'
            elif cx in range(strx+80+80,strx+80+80+80) :
                cv2.rectangle(im, (strx+80+80, stry+126+52+52), (strx+80+80+80, stry+126+52+52+52), (255, 0, 0), -1)    #9
                cflag_button = '9'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126+52+52), (strx+80+80+80+80, stry+126+52+52+52), (255, 0, 0), -1)    #x
                cflag_button = 'x'
        elif cy in range(stry+126+52+52+52,stry+126+52+52+52+52):
            if cx in range(strx,strx+80) :
                cv2.rectangle(im, (strx, stry+126+52+52+52), (strx+80, stry+126+52+52+52+52), (255, 0, 0), -1)   #4
                cflag_button = '4'
            elif cx in range(strx+80,strx+80+80) :
                cv2.rectangle(im, (strx+80, stry+126+52+52+52), (strx+80+80, stry+126+52+52+52+52), (255, 0, 0), -1)   #5
                cflag_button = '5'
            elif cx in range(strx+80+80,strx+80+80+80) :
                cv2.rectangle(im, (strx+80+80, stry+126+52+52+52), (strx+80+80+80, stry+126+52+52+52+52), (255, 0, 0), -1)    #6
                cflag_button = '6'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126+52+52+52), (strx+80+80+80+80, stry+126+52+52+52+52), (255, 0, 0), -1)    #-
                cflag_button = '-'
        elif cy in range(stry+126+52+52+52+52, stry+126+52+52+52+52+52):
            if cx in range(strx,strx+80):
                cv2.rectangle(im, (strx, stry+126+52+52+52+52), (strx+80, stry+126+52+52+52+52+52), (255, 0, 0), -1)   #1
                cflag_button = '1'
            elif cx in range(strx+80,strx+80+80):
                cv2.rectangle(im, (strx+80, stry+126+52+52+52+52), (strx+80+80, stry+126+52+52+52+52+52), (255, 0, 0), -1)   #2
                cflag_button = '2'
            elif cx in range(strx+80+80,strx+80+80+80):
                cv2.rectangle(im, (strx+80+80, stry+126+52+52+52+52), (strx+80+80+80, stry+126+52+52+52+52+52), (255, 0, 0), -1)    #3
                cflag_button = '3'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126+52+52+52+52), (strx+80+80+80+80, stry+126+52+52+52+52+52), (255, 0, 0), -1)    #+
                cflag_button = '+'
        elif cy in range(stry+126+52+52+52+52+52, stry+126+52+52+52+52+52+52):
            if cx in range(strx,strx+80) :
                cv2.rectangle(im, (strx, stry+126+52+52+52+52+52), (strx+80, stry+126+52+52+52+52+52+52), (255, 0, 0), -1)   #+-
                cflag_button = '+-'
            elif cx in range(strx+80,strx+80+80) :
                cv2.rectangle(im, (strx+80, stry+126+52+52+52+52+52), (strx+80+80, stry+126+52+52+52+52+52+52), (255, 0, 0), -1)   #0
                cflag_button = '0'
            elif cx in range(strx+80+80,strx+80+80+80) :
                cv2.rectangle(im, (strx+80+80, stry+126+52+52+52+52+52), (strx+80+80+80, stry+126+52+52+52+52+52+52), (255, 0, 0), -1)   #.
                cflag_button = '.'
            elif cx in range(strx+80+80+80,strx+80+80+80+80) :
                cv2.rectangle(im, (strx+80+80+80, stry+126+52+52+52+52+52), (strx+80+80+80+80, stry+126+52+52+52+52+52+52), (255, 0, 0), -1)   #=
                cflag_button = '='
        else:
            cflag_button=''

        if cflag_button != pflag_button:
            start_time=time.time()
            pflag_button=cflag_button
        else:
            current_time=time.time()
            if current_time-start_time>1.2:
                caltrack= True
                start_time=time.time()

def executeCal(im,flag_button):
    global list_button,ps,cs,caltrack,ans,resulted
    if caltrack is True:
        for i in range (0,list_button.__len__()):
            if flag_button==list_button[i]:
                break
        if i>=0 and i<=9:    #0->9
            if cs == '0' or resulted is True:
                cs = list_button[i]
                resulted = False
            else:
                cs = cs + list_button[i]
        elif i>9 and i <=13:   # +-x/
            if ps!='':
                if ps[-1]=='+' or ps[-1]=='-' or ps[-1]=='x' or ps[-1]=='/':
                    ps = ps[:-1] + list_button[i]
                else:
                    ps = cs + list_button[i]
                    cs = '0'
            else:
                ps = cs + list_button[i]
                cs = '0'
        elif i==15:  #sqrt
            ps=list_button[i]+'('+cs+')'
            calculate('sqrt')
            resulted = True
        elif i==16:  #sqr
            ps=list_button[i]+'('+cs+')'
            calculate('sqr')
            resulted=True
        elif i==17:  #1/
            ps=list_button[i]+'('+cs+')'
            calculate('1/')
            resulted = True
        elif i==18:  #ans
            calculate('ans')
        elif i==19:  #c
            cs = '0'
            ps = ''
            sign_press=False
        elif i==20:  #<-
            if cs.__len__()!=1:
                cs=cs[:-1]
            else:
                cs='0'
        elif i==21: #+-
            calculate('+-')
        elif i==22: #.
            if cs[:-1]!='.':
                cs=cs+list_button[i]
        elif i==23 and ps!='':
            resulted = True
            calculate('=')

        caltrack = False
def calculate(operator):
    global cs,ps,ans,again
    if operator=='sqrt':
        cd=float(cs)
        cs=str(math.sqrt(cd))
        ans=cs
    if operator=='sqr':
        cd=float(cs)
        cs=str(math.pow(cd,2))
        ans = cs
    if operator=='1/x':
        cd=float(cs)
        cs=str(1/cd)
        ans = cs
    if operator=='+-':
        cd=-float(cs)
        cs=str(cd)
        ans=cs
    if operator=='ans':
        cs=ans
    if operator=='=':
        if ps[-1] == '+':
            result = float(cs) + float(ps[:-1])
            cs = str(result)
            ps = ''
        elif ps[-1] == '-':
            result = -float(cs) + float(ps[:-1])
            cs = str(result)
            ps = ''
        elif ps[-1] == 'x':
            result = float(cs) * float(ps[:-1])
            cs = str(result)
            ps = ''
        elif ps[-1] == '/':
            result = float(ps[:-1]) / float(cs)
            cs = str(result)
            ps = ''
        ans=cs
def displayResul(im):
    global cs,ps
    if cs[-2:] == '.0':
        cs=cs[:-2]
    cv2.putText(im, str(cs), (strx + 10, stry + 100), font, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
    if ps != '':
        cv2.putText(im, str(ps), (strx + 10, stry + 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

while True:
    ret,oriFrame=cap.read()
    frame = cv2.flip(oriFrame,1)
    frame = cv2.resize(frame, (1060, 720), interpolation=cv2.INTER_CUBIC)
    if tracked is False:
        for i in range(0, nocheckframe):
            drawCheckFrame(frame, checkframe_pos[i][0], checkframe_pos[i][1], 'g', 1)
        cv2.imshow('Hand Tracking Application',frame)

    else:
        drawframe = frame.copy()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(0,nocheckframe):
            lower,upper=produceBoundarie(profile_hand[i][0],profile_hand[i][1],profile_hand[i][2])
            mask = cv2.inRange(frame_hsv, lower, upper)
            if i==0:
                offmask = mask
            else:
                offmask=mask+offmask

        median = cv2.medianBlur(offmask, 29)

        im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        realHandContour = extractHandContour(contours)

        findHullAndDefects(drawframe, realHandContour)

        realDefect,noDefect=drawDefect(drawframe,realHandContour)

        clickCalculator(drawframe,realHandContour,realDefect,noDefect)

        executeCal(drawframe, cflag_button)
        smallmedian=cv2.pyrDown(median)
        cv2.imshow('mask',smallmedian)
        calROI = drawframe[stry:stry+438, strx:strx+320]  # refractor
        dst = cv2.addWeighted(calc, 0.8, calROI, 0.2, 0)
        drawframe[stry : stry+438, strx:strx+320] = dst
        displayResul(drawframe)
        cv2.imshow('Virtual Calculator',drawframe)
    k = cv2.waitKey(3) & 0xFF


    if k==ord('c'):
        profile_hand = np.zeros(shape=(nocheckframe,3))
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        for i in range (0,nocheckframe):
            profile_hand[i]= np.array(frame_hsv[checkframe_pos[i][1]+5,checkframe_pos[i][0]+5])
        tracked = True
        cv2.destroyWindow('Hand Tracking Application')
    elif k==ord('b'):
        profile_hand = np.zeros(shape=(nocheckframe, 3))
        tracked =False
    elif k == 27:
        break

cv2.destroyAllWindows()
cap.release()
