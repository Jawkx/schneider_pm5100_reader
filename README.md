# Meter Reader

## Introduction and purpose
Started as an intern subproject for Lumileds Penang.

The purpose of this program is to convert large image data to a text data efficiently. 


The input can be seen below

<img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/testing%20images/metertest2.png" width = "480">


## Library used, Language used, Practical implementation 
The code will be written in python 3. With the aid of opencv2 and imultis library.



## Steps taken
### 1.Find device

First, function below is used to find the physical device 

```python
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
```

`dtype` is the type of device, as different device use different variable. The detection of device type will later being integrated into the code using a QR code system.

The function above will return a greyscale image with flatten perspective as shown as below.

Input:


