import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
tracked = False
checkframe_pos = np.array([[420, 120], [420, 260], [470, 150], [440, 320], [390, 210], [460, 230], [480, 250]])
nocheckframe=checkframe_pos.__len__()
font = cv2.FONT_HERSHEY_SIMPLEX
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
    angleTol=75
    tolerance=h/5
    nodefects=0
    realDefect=defects.copy()
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(realHandContour[s][0])
        end   = tuple(realHandContour[e][0])
        far   = tuple(realHandContour[f][0])
        if ((distanceP2P(start[0],start[1],far[0],far[1])>tolerance) and (distanceP2P(end[0],end[1],far[0],far[1])>tolerance) and (getAngle(start[0],start[1],far[0],far[1],end[0],end[1]) < angleTol)):
            cv2.circle(im,start,5,(255,0,0),2)
            cv2.putText(im,'%d' %(i), (start),font,1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(im, far, 5, (0, 255, 0), 2)
            cv2.putText(im, '%d' % (i), (far), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(im, end, 5, (0, 0, 255), 2)
            cv2.putText(im, '%d' % (i), (end), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            nodefects=nodefects+1
    duc='Number of fingcer: %d' %(nodefects)
    cv2.putText(im, duc, (20,20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
def check_check(des,cx,cy):
    if cx in range(420,430):
        if cy in range (260,270):
            drawCheckFrame(des,420,260,'g',-1)
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
    frame=cv2.flip(oriFrame,1)
    for i in range(0,nocheckframe):
        drawCheckFrame(frame,checkframe_pos[i][0],checkframe_pos[i][1],'r',1)
    if tracked is False:
        cv2.imshow('Flipped',frame)
    else:
        drawframe=frame.copy()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(0,nocheckframe):
            lower,upper=produceBoundarie(profile_hand[i][0],profile_hand[i][1],profile_hand[i][2])
            mask = cv2.inRange(frame_hsv, lower, upper)
            if (i==0):
                offmask = mask
            else:
                offmask=mask+offmask

        median = cv2.medianBlur(offmask, 29)
        im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        a = extractHandContour(contours)
        #cx,cy=drawCenter(drawframe,a)
        #check_check(drawframe,cx,cy)
        findHullAndDefects(drawframe, a)
        drawDefect(drawframe,a)
        cv2.imshow('mask',median)
        cv2.imshow('draw',drawframe)
    k = cv2.waitKey(5) & 0xFF


    if k==ord('c'):
        profile_hand=a = np.zeros(shape=(7,3))
        max,min=0,255
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        for i in range (0,nocheckframe):
            profile_hand[i]= np.array(frame_hsv[checkframe_pos[i][1]+5,checkframe_pos[i][0]+5])

        tracked = True


    elif k == 27:
        break

cv2.destroyAllWindows()
cap.release()
