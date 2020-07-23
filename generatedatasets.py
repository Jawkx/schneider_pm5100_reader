from functions import *
import os
img = cv2.imread("testing_images/metertest2.png")
imgarr = []

for idx in range()
device = findDevice(img,type1)
screen = findScreen(device,type1)

count = 0 
textblocks, dotCoor = extractTextImg(screen,type1)
for textblock in textblocks:
    digits , null = extractDigit(textblock,30,type1)
    for digit in digits:
        cv2.imshow("digit",digit)
        k = cv2.waitKey(0)
        if k != 32:
            print( chr(k) )
            cv2.imwrite( "./trainingdatasets/" + chr(k) + "__" + str(count)+".png" , digit)
            count + = 1
