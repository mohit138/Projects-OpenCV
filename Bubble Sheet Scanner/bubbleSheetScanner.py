import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours

image = cv2.imread("omr1.png")
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0) 
edged = cv2.Canny(blurred, 75, 200)

cv2.imshow("edge",edged)

# find the contour for the sheet
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt= None

if len(cnts)>0:
    
    cnts = sorted(cnts , key=cv2.contourArea, reverse =True)
    print("found")
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*.02, True)
        #cpy = image.copy()
        #cv2.drawContours(cpy, [approx], -1,(0,255,0),2)
        #cv2.imshow("cnt", cpy)
        #cv2.waitKey(0)
        
        if len(approx) == 4:
            docCnt = approx
            break

#cv2.drawContours(image, [docCnt], -1,(255,0,0),2)
#cv2.imshow("cnt", image)
#cv2.waitKey(0)


# perform perspective transform
# reshaping docCnt from 4,1,2 to 4,2
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

cv2.imshow("paper", paper)
cv2.imshow("gpaper", warped)
cv2.waitKey(0)

# apply threshold
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                                                                        
cv2.imshow("thresh", thresh)
cv2.waitKey()


# DETECTING BUBBLES FOR EACH QUESTIONS
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    # now check if box is sufficiently big, and is a square
    if w>=20 and h>=20 and ar>=.9 and ar<=1.1:
        questionCnts.append(c)
        cpy = warped.copy()
        cv2.drawContours(cpy, [c], -1,(0,255,0),2)
        cv2.imshow("cnt", cpy)
        cv2.waitKey(0)
        
# now we will perform the grading part on our OMR

        
    




cv2.destroyAllWindows()