import cv2
import numpy as np


dtype1_lbl = ["V avg","I avg","P tot","E del"]

#loads template
#type1 
temptype1 = []

for x in range(0,9):
    temp = cv2.imread("./templates/"+ str(x) +".PNG" , cv2.IMREAD_GRAYSCALE)
    temptype1.append(temp)


def findDevice(img,dtype):
    if dtype == 0:
        threshold = 40
        out = np.float32([[0,300],[0,0],[300,0],[300,300]])

    im_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im_grey, threshold, 255, cv2.THRESH_BINARY_INV)

    allcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    devicecontours = max(allcontours, key = cv2.contourArea)
    rect = cv2.minAreaRect(devicecontours)
    box = cv2.boxPoints(rect).astype("int")
    
    M = cv2.getPerspectiveTransform(np.float32(box),out)
    return cv2.warpPerspective(im_grey,M,(300,300))

    
def findScreen(device,dtype):
    if dtype == 0:
        threshold = 80
        out = np.float32([[0,300],[0,0],[300,0],[300,300]])
        
    ret,thresh = cv2.threshold(device, threshold, 255, cv2.THRESH_BINARY)
    screencontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if screencontours!= None:
        screen = max(screencontours, key = cv2.contourArea)
        rect = cv2.minAreaRect(screen)
        box2 = cv2.boxPoints(rect).astype("int")
        M = cv2.getPerspectiveTransform(np.float32(box2),out)
        screen = cv2.warpPerspective(device,M,(300,300))
    else:
     print("ERROR CAN'T FIND SCREEN")
     
    cv2.imshow("press spacebar to rotate, any to continue",screen)
    k = cv2.waitKey(0)
    if k == 32:
        screen = cv2.rotate(screen, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
    cv2.destroyAllWindows()
    return screen


def extractTextImg(screen,dtype):
    if dtype == 0:
        ratiobase = 25
        morphsize = (5,1)
        dilation_no = 3
        maxcount = 4
    
    croppedlist = []
    threshold = 255
    while True:
        ret,thresh = cv2.threshold(screen, threshold, 255, cv2.THRESH_BINARY_INV)
        x,y = thresh.shape
        whitecount = cv2.countNonZero(thresh)
        blackcount = (x*y) - whitecount
        ratio = (whitecount/(blackcount+whitecount))*100
        threshold = threshold - 17
        if ratio < ratiobase:
            break;
            
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphsize) 
    dilation = cv2.dilate(thresh, rect_kernel, iterations = dilation_no) 
    textcontours, texthierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for textc in textcontours:
    
        x, y, w, h = cv2.boundingRect(textc)
        textsize = w*h

        
        if textsize > 2000 and textsize < 8000:
            count = count+1
            cv2.putText(screen,str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,1,cv2.LINE_AA)
            rect = cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 2)
            croppedtext = thresh[y:y + h, x:x + w] 
            croppedlist.append(croppedtext)
            
    croppedlist = croppedlist[0:maxcount] 
    croppedlist.reverse()

        
    if count >= maxcount:
        return croppedlist
    else:
        print(count)
        print ("CAN'T FIND ENOUGH NUMBER")


def readNumb(numbpic,dtype):
    w,h = temptype1[2].shape
    res = cv2.matchTemplate(numbpic,temptype1[2], cv2.TM_CCOEFF )
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(numbpic,top_left, bottom_right, 255, 2)
    
    return numbpic
  
im = cv2.imread("metertest5.png")
device = findDevice(im,0)
screen = findScreen(device,0)
cropped = extractTextImg(screen,0)
num = readNumb(cropped[2],1)
cv2.imshow("im",num)

'''
for idx,cropimg in enumerate(cropped): 

    cv2.imshow(dtype1_lbl[idx],cropimg)
    cv2.waitKey(0) 
'''

cv2.waitKey(0) 