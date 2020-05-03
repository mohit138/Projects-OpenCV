# this file is used to scan documents
# we will identify the dc, crop it and apply perspective transform !

from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

image = cv2.imread("3.jpg")
image = imutils.resize(image,height=500)
orig = image.copy()

# first step is to find edges

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2. GaussianBlur(gray , (3,3),0)
#cv2.imshow("blur", gray)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("Step I - egde detection", edged)
cv2.waitKey(0)
# second step is to find contours
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort in descending order and select first 5
cnts = sorted(cnts , key = cv2.contourArea, reverse=True)[:5]



for c in cnts:
    # approx the contour
    peri = cv2.arcLength(c,True)
    # pass 1-5 percent of perimeter for epsilon
    approx = cv2.approxPolyDP(c, peri*.02, True)
    #cv2.drawContours(image, [c], -1, (0,255,0), 2)
    #cv2.imshow("c",image)
    #cv2.waitKey(0)
    # if the approximated contour has 4 points, then we have our rectangle
    # screen
    if len(approx) == 4:
        screenCnt = approx
        break
    

cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
cv2.imshow("Step II - detected contour",image)
cv2.waitKey(0)


# step 3 , apply perspective transform
#screenCnt has shape (4,1,2) , hence we need to reshape it before sendong 
[warped,w,h] = four_point_transform(orig, screenCnt.reshape(4,2))
if h<w:
    warped = imutils.rotate_bound(warped, -90)

# convert im to gray and then threshold it
# to give black and white threshold effect 
cv2.imshow("Document",imutils.resize(warped, height = 600))
cv2.waitKey(0)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

#for i in range(15):
T = threshold_local(warped,13,offset=7,method='gaussian')
Warped =(warped > T).astype("uint8")*255

cv2.imshow("Step III - applying perspective transform",imutils.resize(Warped, height = 600))
cv2.waitKey(0)

cv2.destroyAllWindows()