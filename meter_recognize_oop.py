import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

dtype1_lbl = ["V avg","I avg","P tot","E del"]

#loads template
#type1 
temptype1 = []

for x in range(0,10):
    temp = cv2.imread("templates/"+ str(x) +".png" ,0)
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

def extractDigit(numblock,dtype):
    cv2.imshow("show",numblock)
    hblock,wblock = numblock.shape
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10)) 
    dilation = cv2.dilate(numblock, rect_kernel, iterations = 2) 
    cnts = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    extractedcount = 0
    digitimg = []
    for c in cnts:
        extractedcount = extractedcount + 1
        (x, y, w, h) = cv2.boundingRect(c)
        if h > hblock*0.7:
            digit = numblock[y:y + h, x:x + w]
            digitimg.append(digit)

    digitimg.reverse()
    return digitimg
    
def recognizeDigit(digit,dtype):

    h,w = digit.shape
    error = []
    total = 0
    
    for idx,current in enumerate(temptype1): 
        temps = cv2.resize(temptype1[idx], (w,h) , interpolation = cv2.INTER_AREA)
        difference = cv2.countNonZero( digit - temps )
        total = total + difference
        error.append(difference)
        
    average = total / 10
    val, idx = min((val, idx) for (idx, val) in enumerate(error))
    confidence = round((average - val)/average , 3)*100
    
    return confidence , idx

    
im = cv2.imread("testing images/metertest7.png")
device = findDevice(im,0)
screen = findScreen(device,0)
textblock = extractTextImg(screen,0)
digits = extractDigit(textblock[1],0)
confidence , digit = recognizeDigit(digits[2] , 1)


print(digit)

