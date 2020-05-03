# this file used to crop rectangle with 4 pints and apply perspective transform 
import numpy as np
import cv2

def order_points(pts):
    # we will take 4 pints and set them  in an ordered manner. 
    # such that - top left, top right, bottom right , bottom left 
    # -> 1,2,3,4
    rect = np.zeros((4,2),dtype="float32")
    
    s = pts.sum(axis=1)
    # top left -> least sum 
    rect[0] = pts[np.argmin(s)]
    # bottom roght -> max sum
    rect[2] = pts[np.argmax(s)]
    
    
    diff = np.diff(pts,axis=1) # y-x 
    # top right ->smallest diff
    rect[1] = pts[np.argmin(diff)]
    # bottom left -> largest diff
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image,pts):
    
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    # now we need to obtain main width and height which is possible
    # find width of both upper and lower sides
    widthA =np.sqrt(((br[0]-bl[0])**2) + ((br[1] - bl[1])**2))
    widthB =np.sqrt(((tr[0]-tl[0])**2) + ((tr[1] - tl[1])**2))
    # now we consider the max width
    maxWidth = max(int(widthA), int(widthB))
    
    
    #similarly find the max height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now lets define the dimentions for the new image.
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth -1 , maxHeight -1],
        [0, maxHeight-1]] , dtype ="float32")
    
    # compute the perspective transform, 
    M =cv2.getPerspectiveTransform(rect, dst)
    
    # apply this transform matrix on the image
    warped = cv2.warpPerspective(image, M, (maxWidth,maxHeight))
    
    # return the warpped image
    return [warped,maxWidth,maxHeight]

    
    
    