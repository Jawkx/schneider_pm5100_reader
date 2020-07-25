import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from copy import copy


temptype1 = []
for i in range(0, 10):
    temp = cv2.imread("templates/"+ str(i) +".png", 0)
    temptype1.append(temp)

param = {
    "deviceThresh" : 40,
    "screenThresh" : 80,
    "outpar" : np.float32([[0, 300], [0, 0], [300, 0], [300, 300]]),
    "outsize" : (300, 300),
    "ratiobase" : 28,
    "morphsize" : (6, 1),
    "dilation_no" : 3,
    "maxcount" : 4,
    "temps" : temptype1,
    "digitmorphsize" : (1, 8),
    "digit_dilation_no" : 2,
    "label" : ["V avg", "I avg", "P tot", "E del"],
    "unit" :[" V", " A", " kw", " Gwh"],
    "dighighratio" : 0.85
}

def debugWindow(target):
    cv2.imshow("debugwindow",target)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if k==113:
        exit()

#######
def rotateimg(img):
    rotated = img
    x,y = img.shape
    center = ( 0 , int(y/2) )
    while True:
        rotatedShow = copy(rotated)
        cv2.putText(rotatedShow,"Press R to rotate",center,cv2.FONT_HERSHEY_SIMPLEX ,1,0,2)
        cv2.putText(rotatedShow,"Space to continue",(0, int(y/2) + 30),cv2.FONT_HERSHEY_SIMPLEX ,1,0,2)
        cv2.imshow("Rotate?", rotatedShow)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k in (114, 82):
            rotated = cv2.rotate(rotated, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif k == 32:
            break

    return rotated

def findDevice(img):
    threshold = param["deviceThresh"]
    out = param["outpar"]
    im_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im_grey, threshold, 255, cv2.THRESH_BINARY_INV)

    allcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    devicecontours = max(allcontours, key = cv2.contourArea)
    rect = cv2.minAreaRect(devicecontours)
    box = cv2.boxPoints(rect).astype("int")

    M = cv2.getPerspectiveTransform(np.float32(box),out)
    return cv2.warpPerspective(im_grey,M,param["outsize"])

def findScreen(device):
    threshold = param["screenThresh"]
    out = param["outpar"]

    ret,thresh = cv2.threshold(device, threshold, 255, cv2.THRESH_BINARY)
    screencontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if screencontours!= None:
        screen = max(screencontours, key = cv2.contourArea)
        rect = cv2.minAreaRect(screen)
        box2 = cv2.boxPoints(rect).astype("int")
        M = cv2.getPerspectiveTransform(np.float32(box2),out)
        screen = cv2.warpPerspective(device,M,param["outsize"])
    else:
        print("ERROR CAN'T FIND SCREEN")

    screen = rotateimg(screen)

    return screen

def findDot(img):
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.erode(img, kernel, iterations=10)
    cnts,hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dotCnt = min(cnts,key= cv2.contourArea)
    (x,y),radius = cv2.minEnclosingCircle(dotCnt)
    if radius > 3 or radius < 1:
        print("ERROR:CAN'T FIND DOT,INPUT DECIMAL POINT MANAULLY")
        cv2.imshow("Point decimal point" , img)
        k = cv2.waitKey(0)
        x = int(chr(k))
        noDot = True
        return x,y , int(radius) , noDot
    else: 
        noDot = False
        return x,y , int(radius) ,noDot
    
def extractTextImg(screen):
    scrw,scrh = screen.shape
    ratiobase = param["ratiobase"]
    morphsize = param["morphsize"]
    dilation_no = param["dilation_no"]
    maxcount = param["maxcount"]

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
            break

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphsize)
    dilation = cv2.dilate(thresh, rect_kernel, iterations = dilation_no)
    textcontours, texthierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dotX = []
    noDotarr = []
    for textc in textcontours:
        x, y, w, h = cv2.boundingRect(textc)
        textsize = w*h

        if textsize > 2000  and w < 200:
            rect = cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 1)
            croppedtext = thresh[y:y + h, x:x + w]
            x,y,radius,noDot = findDot(croppedtext)
            if not noDot:
                cv2.circle(croppedtext,( int(x),int(y) ),radius,0,-1)
            croppedlist.append(croppedtext)
            dotX.append(x)
            if noDot:
                noDotarr.append(True)
            else:
                noDotarr.append(0)

    noDotarr = noDotarr[0:maxcount]
    dotX = dotX[0:maxcount]
    croppedlist = croppedlist[0:maxcount]
    croppedlist.reverse()
    dotX.reverse()
    noDotarr.reverse()

    return croppedlist , dotX , noDotarr

def findDotDecimalPos(digitsCoor,dotCoor,noDot):
    if noDot:
        return dotCoor

    digitsCoor.append(10000)
    for idx,digitCoor in enumerate(digitsCoor):
        if dotCoor <= digitCoor:
            return idx 
    
def extractDigit(numblock,dotPos,noDot):
    hblock,wblock = numblock.shape

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, param["digitmorphsize"])
    dilation = cv2.dilate(numblock, rect_kernel, iterations = param["digit_dilation_no"])
    cnts = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    extractedcount = 0
    digitimg = []
    digitimgCoor = []
    for c in cnts:
        extractedcount = extractedcount + 1
        (x, y, w, h) = cv2.boundingRect(c)
        if h > hblock*param["dighighratio"]:
            digit = numblock[y:y + h, x:x + w]
            digitimg.append(digit)
            digitimgCoor.append(x)
    
    digitimg.reverse()
    digitimgCoor.reverse()
    dotPos = findDotDecimalPos(digitimgCoor,dotPos,noDot)
    return digitimg , dotPos

def recognizeDigit(digit):
    temptype = param["temps"]
    h,w = digit.shape
    error = []
    total = 0

    for idx,current in enumerate(temptype):
        temps = cv2.resize(temptype[idx], (w,h) , interpolation = cv2.INTER_AREA)
        difference = cv2.countNonZero( digit - temps )
        total = total + difference
        error.append(difference)

    average = total / 10
    val, idx = min((val, idx) for (idx, val) in enumerate(error))
    confidence = round((average - val)/average , 3)*100

    if confidence < 25:
        print("\nLOW confidence, input digit manually")
        cv2.imshow("x", digit)
        k = cv2.waitKey(0)
        idx = chr(k)
        print ("Inputted digit: " + idx)
        confidence = 100
    else:
        idx = str(idx)

    return idx , confidence

def digits2string(digits,dchar):
    digitLst = []
    confidenceLst = []
    for digit in digits:
        number ,confidence = recognizeDigit(digit)
        digitLst =  digitLst + [number]
        confidenceLst = confidenceLst + [confidence]

    charlst = ''.join(digitLst[:dchar])
    manlst = ''.join(digitLst[dchar:])

    out = charlst + '.' + manlst
    return float(out)

class meter:
    def __init__(self,image):
        self.blank = 255 * np.ones(shape=[300, 300, 1])
        self.im = cv2.imread(image)
        self.meterlabel = param["label"]

        self.device = findDevice(self.im)
        self.screen = findScreen(self.device)
        self.textblocks, self.dotCoor , self.noDotArr= extractTextImg(self.screen)

        self.numbers = []

        for idx,textblock in enumerate(self.textblocks):
            digits,dotPos = extractDigit(textblock,self.dotCoor[idx],self.noDotArr[idx])
            digitWithDecimal = digits2string(digits,dotPos)
            self.numbers.append(digitWithDecimal)
            cv2.putText(self.blank, param["label"][idx] + " = " + str(digitWithDecimal) + param["unit"][idx], (0, int(idx * 300/param["maxcount"])+30),cv2.FONT_HERSHEY_SIMPLEX ,0.7,0,2)

    def printLabel(self):
        for idx in range(self.param["maxcount"]):
            print(self.param["label"][idx] + " = ", self.numbers[idx] , self.param["unit"][idx])
            
    def check(self):
        cv2.imshow("screen",self.screen)
        cv2.imshow("value",self.blank)
        cv2.waitKey(0)

    def debugPrompt(self):
        cv2.imshow("device", self.device)
        cv2.imshow("screen", self.screen)
        for idx,textblock in enumerate(self.textblocks):
            cv2.imshow("textblock" + str(idx) , textblock)
        cv2.waitKey(0)

