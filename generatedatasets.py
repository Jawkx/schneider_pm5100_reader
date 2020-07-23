from functions import *
import os
#img = cv2.imread("testing_images/metertest2.png")
imgarr = []
count = 0 
for idx in range(0,12):
    img = cv2.imread("testing_images/metertest" + str(idx) + ".png")
    imgarr.append(img)

for img in imgarr:
    device = findDevice(img,type1)
    screen = findScreen(device,type1)
    
    textblocks, dotCoor = extractTextImg(screen,type1)
    for textblock in textblocks:
        digits , null = extractDigit(textblock,30,type1)
        for digit in digits:
            cv2.imshow("digit",digit)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k != 32:
                print( chr(k) )
                cv2.imwrite( "./training_data/" + str(count) + "_" + chr(k) +".png" , digit)
                count += 1
