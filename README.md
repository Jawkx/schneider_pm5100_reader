# Meter Reader

## Introduction and purpose
This project is a solo project when im working as an intern at Lumileds Penang to aid with automation of data collecting for various equipment.

The purpose of this program is to convert large amount of image data to a text data quickly and accurately. 

The goal is to achieve action as table below

Input | Output
--- | ---
<img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/testing%20images/metertest2.png" width = "300"> | <img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/testing%20images/metertest2.png" width = "300">


## Library used, Language used, Practical implementation 
The code will be written in [Python 3](https://www.python.org/). 

Image processing was done with the aid of [Open CV 2](https://opencv.org/) and [imultis](https://github.com/jrosebr1/imutils) librarys

Data processing and integration with excel sheet was done with [Pandas](https://pandas.pydata.org/) Library



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

Input | Output
--- | ---
<img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/testing%20images/metertest2.png" width = "200"> | <img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/documentation%20pics/deviceimg.png" width = "200">
