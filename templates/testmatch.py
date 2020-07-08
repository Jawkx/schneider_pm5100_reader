import numpy as np
import cv2
import matplotlib.pyplot as plt

temps = []

    
img = cv2.imread('test3.png',0)
h,w = img.shape
for x in range(0,9):
    temp = cv2.imread( str(x) + '.png' ,0)
    resizedtemp = cv2.resize(temp, (w,h) , interpolation = cv2.INTER_AREA)
    temps.append(resizedtemp)

error = [] 
total = 0
for idx,current in enumerate(temps): 
    difference = cv2.countNonZero( img - temps[idx])
    total = total + difference
    error.append(difference)
    
average = total/10

val, idx = min((val, idx) for (idx, val) in enumerate(error))
confidence = round((average - val)/average , 3)

print(confidence)
print(idx)

#cv2.imshow('img',test)

#cv2.waitKey(0)