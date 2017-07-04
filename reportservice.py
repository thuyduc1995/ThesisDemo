import cv2
import numpy as np
import math

img = cv2.imread('filter.jpg',1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('vedcroi.jpg',1)

ret,thresh = cv2.threshold(img_gray,25,255,cv2.THRESH_BINARY)
#cv2.imwrite('vedcroi.jpg',thresh)

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
cv2.drawContours(img2, contours, -1, (0,255,0), 3)
cv2.drawContours(img, contours, -1, (0,255,255), 3)
cv2.imshow('threshold', img)


k = cv2.waitKey(0)

cv2.destroyAllWindows()