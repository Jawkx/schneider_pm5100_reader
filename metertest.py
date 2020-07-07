import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

imageLink = "metertest5.png"
displaytype = 0


#type 1 Display  
def drawdot(boxinp,target,color):

    for point in boxinp:
       x = point[0]
       y = point[1]
       coor = (x,y)
       if color == 1:
         cv2.circle(target,(x,y),5,(0, 0, 255), -1)
       else:
         cv2.circle(target,(x,y),5,(255, 255, 255), -1)
       

out = np.float32([[0,300],[0,0],[300,0],[300,300]])


##Find device
threshold = 40
im = cv2.imread(imageLink)
w,h,c = im.shape
im_grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(im_grey, threshold, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect).astype("int")
drawdot(box,im,1)
M = cv2.getPerspectiveTransform(np.float32(box),out)
#cv2.drawContours(im, [box], -1, (0,255,0), 1)
device = cv2.warpPerspective(im_grey,M,(300,300))

#Find screen
threshold2 = 80
ret,thresh2 = cv2.threshold(device, threshold2, 255, cv2.THRESH_BINARY)
contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c2 = max(contours2, key = cv2.contourArea)
rect2 = cv2.minAreaRect(c2)
box2 = cv2.boxPoints(rect2).astype("int")
M2 = cv2.getPerspectiveTransform(np.float32(box2),out)
screen = cv2.warpPerspective(device,M2,(300,300))

#ask user whether rotate or not
cv2.imshow("rotate? press y/n",screen)
k = cv2.waitKey(0)
cv2.destroyAllWindows() 

if k == 121 : #yes
    print("Rotated")
    screen = cv2.rotate(screen, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 


threshold3 = 255
while True:
    ret,thresh3 = cv2.threshold(screen, threshold3, 255, cv2.THRESH_BINARY_INV)
    x,y = thresh3.shape
    whitecount = cv2.countNonZero(thresh3)
    blackcount = (x*y) - whitecount
    ratio = (whitecount/(blackcount+whitecount))*100
    threshold3 = threshold3 - 17
    if ratio < 25:
        break;
    
    
print(threshold3)
print(ratio)

#get text contours

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)) 
dilation = cv2.dilate(thresh3, rect_kernel, iterations = 3) 
textcontours, texthierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(screen, textcontours, -1, (0,255,0), 1)

count = 0 
for textc in textcontours:
    
    x, y, w, h = cv2.boundingRect(textc)
    textsize = w*h 
    if textsize > 2000 and textsize < 8000:
        count = count+1
        cv2.putText(screen,str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,1,cv2.LINE_AA)
        rect = cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 2)
        croppedtext = screen[y:y + h, x:x + w] 
        cv2.imshow("cropped %d"+str(count),croppedtext)

        

##Display
outputvar = screen
cv2.imshow('metertest', outputvar )



cv2.waitKey(0)

cv2.destroyAllWindows()