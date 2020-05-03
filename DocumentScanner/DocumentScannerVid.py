# this file is used to scan documents
# we will identify the dc, crop it and apply perspective transform !

from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils



cap=cv2.VideoCapture("vid.mp4")
w=cap.get(3)
h=cap.get(4)
print("{}    {}".format(w,h))
w=int(.3*w)
h=int(.3*h)
print("{}    {}".format(w,h))

Warped=np.zeros((500,373))

if (cap.isOpened()==False):
    print("Error opening video file")
    
while(cap.isOpened()):
    
    
    
    ret, frame= cap.read()
    frame = cv2.resize(frame,(w,h))    
    
    if ret == True:
        #frame = cv2.resize(frame,(w,h))    
        cv2.imshow("frame",imutils.resize(frame, height = 350))
        
        orig = frame.copy()
        fcpy= frame.copy()
        # first step is to find edges

        gray = cv2.cvtColor(fcpy, cv2.COLOR_BGR2GRAY)
        gray = cv2. GaussianBlur(gray , (7,7),0)
        #cv2.imshow("blur", gray)
        edged = cv2.Canny(gray, 95, 200)
        
        #cv2.imshow("Step I - egde detection", edged)
        #cv2.waitKey(0)

        # second step is to find contours
        cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # sort in descending order and select first 5
        cnts = sorted(cnts , key = cv2.contourArea, reverse=True)[:5]
        
        flag =0 # here if doc is not detected
        
        
        for c in cnts:
            # approx the contour
            peri = cv2.arcLength(c,True)
            # pass 1-5 percent of perimeter for epsilon
            approx = cv2.approxPolyDP(c, peri*.02, True)
            #cv2.drawContours(frame, [c], -1, (0,255,0), 2)
            #cv2.imshow("c",frame)
            #print(len(approx))
            #cv2.waitKey(0)
            # if the approximated contour has 4 points, then we have our rectangle
            # screen
            if len(approx) == 4:
                screenCnt = approx
                flag =1
                break
            
        if flag == 0:
            print("UN-DETECTED")
        else :
            print("DETECTED")
            #cv2.drawContours(fcpy, [screenCnt], -1, (0,255,0), 2)
            #cv2.imshow("Step II - detected contour",fcpy)
            #cv2.waitKey(0)
            
            
            # step 3 , apply perspective transform
            #screenCnt has shape (4,1,2) , hence we need to reshape it before sendong 
            [warped,W,H] = four_point_transform(orig, screenCnt.reshape(4,2))
            if H<W:
                warped = imutils.rotate_bound(warped, 90)
                
            # convert im to gray and then threshold it
            # to give black and white threshold effect 
            #cv2.imshow("Document",imutils.resize(warped, height = 600))
            #cv2.waitKey(0)
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            #for i in range(15):
            T = threshold_local(warped,13,offset=7,method='gaussian')
            Warped =(warped > T).astype("uint8")*255
            
        cv2.imshow("Step III - applying perspective transform",imutils.resize(Warped, height = 600))
            #cv2.waitKey(0)
            
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

