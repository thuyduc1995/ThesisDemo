import cv2
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
tracked = False
checkframe_pos = np.array([[600, 120], [600, 260], [650, 150], [620, 320], [570, 210], [640, 230], [660, 250]])
nocheckframe=checkframe_pos.__len__()
font = cv2.FONT_HERSHEY_SIMPLEX
b_point=[]
g_point=[]
r_point=[]
flag_color=''
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
def drawTool(im,realHandContour,realDefect,nodefect):
    global b_point,g_point,r_point,flag_color
    if (nodefect>0):
        listEndPoint=[]
        for i in range (0,nodefect):
            listEndPoint.append(realDefect[i,0,1])
        listEndPoint.sort()

        end = tuple(realHandContour[listEndPoint[0]][0])
        cx =end[0]
        cy =end[1]
        if cy in range(0, 100):
            if cx in range (400, 550):
                flag_color='b'
            if cx in range(580, 730):
                flag_color='g'
            if cx in range(760, 910):
                flag_color='r'
        else:
            if flag_color=='b':
                b_point.append([cx, cy])
            if flag_color=='g':
                g_point.append([cx, cy])
            if flag_color=='r':
                r_point.append([cx, cy])

    for i in range (0,b_point.__len__()):
        cv2.circle(im,(b_point[i][0],b_point[i][1]),10,(255,0,0),-1)
    for i in range (0,g_point.__len__()):
        cv2.circle(im,(g_point[i][0],g_point[i][1]),10,(0,255,0),-1)
    for i in range (0,r_point.__len__()):
        cv2.circle(im,(r_point[i][0],r_point[i][1]),10,(0,0,255),-1)

while True:
    ret,oriFrame=cap.read()
    frame = cv2.flip(oriFrame,1)
    frame = cv2.resize(frame, (1160, 720), interpolation=cv2.INTER_CUBIC)
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
            if i==0:
                offmask = mask
            else:
                offmask=mask+offmask

        median = cv2.medianBlur(offmask, 29)
        median[400:720,0:300]=np.zeros(shape=(320,300))
        im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        realHandContour = extractHandContour(contours)
        findHullAndDefects(drawframe, realHandContour)

        realDefect,noDefect=drawDefect(drawframe,realHandContour)

        cv2.rectangle(drawframe, (400, 0), (550, 100), (255,0,0), -1)
        cv2.rectangle(drawframe, (580, 0), (730, 100), (0,255,0), -1)
        cv2.rectangle(drawframe, (760, 0), (910, 100), (0,0,255), -1)

        drawTool(drawframe, realHandContour, realDefect, noDefect)
        smallmedian=cv2.pyrDown(median)
        cv2.imshow('mask',smallmedian)

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
