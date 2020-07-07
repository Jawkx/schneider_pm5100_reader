import os
import cv2 
imgsdir = os.listdir('templates')


for imgdir in imgsdir:
    
    #print(imgdir)
    im = cv2.imread('templates/' + imgdir , cv2.IMREAD_GRAYSCALE)
    w,h = im.shape
    print(w,h, "---")
    cropped = im[int(h/3):int(3*h/2) , 0:w]
    
    w,h = cropped.shape
    print(w,h, "-")
    cv2.imshow("im",cropped)
    cv2.waitKey(0)