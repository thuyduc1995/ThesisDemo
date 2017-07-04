import cv2
import time
cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
duc=100
start=time.time()
while True:
    im,frame=cap.read()

    duc=duc-1
    if (duc==0):
        end=time.time()
        print end-start
        duc=100
    #cv2.putText(frame, '%d' % (fps), (10,10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    k = cv2.waitKey(3) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()