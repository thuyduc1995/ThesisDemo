import numpy as np
import cv2

#create a VideoCapture object
cap = cv2.VideoCapture(0)

#font = cv2.FONT_HERSHEY_SIMPLEX
#def drawcal(a,b,z,k):
#    cv2.rectangle(k, (a, b), (a+30, b+30), (255, 255, 0), 1)
#    cv2.putText(k, z , (a+10, b+20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame

    ret, thresh = cv2.threshold(gray, 70, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
    #drawcal(400,100,'7',frame)

    cv2.imshow('original',frame)
    cv2.imshow('contour',gray)
    cv2.imshow('mask',thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()