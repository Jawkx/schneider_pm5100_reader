from functions import *
import os
#done till 19
imgarr = []
count = 519 
for idx in range(20,30):
    print("picture" + str(idx))
    img = cv2.imread("testing_images/metertest" + str(idx) + ".png")
    imgarr.append(img)

for img in imgarr:
    device = findDevice(img)
    screen = findScreen(device)
    
    textblocks, dotCoor , nodot= extractTextImg(screen)
    for textblock in textblocks:
        digits , null = extractDigit(textblock,70,False)
        for digit in digits:
            cv2.imshow("digit",digit)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k != 32:
                print( count ,chr(k) )
                cv2.imwrite( "./datasets/" + str(count) + "_" + chr(k) +".png" , digit)
                count += 1
