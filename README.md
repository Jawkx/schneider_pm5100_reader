# Schneider pm5100 Meter Reader
## Introduction and purpose
This project is a solo project when im working as an intern at Lumileds Penang to aid with automation of data collecting for various equipment.

The purpose of this program read many meter image and extract it's reading quickly. And write it into an excel sheet 

The end result is as table below

Input | Output
--- | ---
<img src = "https://github.com/Jawkx/schneider_pm5100_reader/blob/master/testing_images/metertest1.png" width = "300"> | <img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/documentation%20pics/Excel%20output.png" width = "300">


## Library used, Language used, Practical implementation 
The code will be written in [Python 3](https://www.python.org/). 

Image processing was done with the aid of [Open CV 2](https://opencv.org/) ,[imultis](https://github.com/jrosebr1/imutils) and [matplotlib](https://matplotlib.org/)librarys

Data processing and integration with excel sheet was done with [Pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) Library  

## Installation
### 1. Install Python 3
Head to the python website and download [Python 3](https://www.python.org/). It is recommend to download the version after 3.0.0
Remember tick the add to path button as image below:

<img src = "https://datatofish.com/wp-content/uploads/2018/10/0001_add_Python_to_Path.png" width = 300>

All the other setting can use default setting. To check if python was installed correctly type `python --version` into the cmd console. `python (version number)` will be returned.

### 2.Install all required library
Then check if [pip](https://pypi.org/project/pip/) was installed in the computer or not. It should be installed together with python. Pip was a package installer to let you install python addon/library easily in the command line. You can check if pip is installed by pressing `pip --version` into the command line. It will return pip version number and the location in the computer.

To install all library for this to work on your computer.Type the below command into command line

```
pip install numpy
pip install pandas
pip install opencv-python
pip install matplotlib
pip install imultis
```
To test if everything is installed correctly. Type `python` into the command line. There should be visual such as `>>` in the command line. Then type code below and see if there is any error.

``` python
import numpy
import pandas
import cv2
import matplotlib
import imultis
```
If there is nothing happen the installation is completed.

## Using the application
Inside the enviroment, create a new object with the class name meter, the parameters put in the source of the image that contain the meter and the type of the meter. Example as below:

```python
meter1 = meter("testing images/metertest7.png",type1)
```

When an object with `meter` class was created, a image of a screen will pop out. if the orientation was wrong press `R` or `r` to rotate. After desirable orientation was presented press `spacebar`

This action will then pop out a digit as shown below. The digit will pop out when the program have low confidence of the recognition of the digit. Press the digit that was shown `(0-9 keys)`. Repeat the process if there is another digit.

The class meter have method as below:
Method | Funtion
---|---
`.printLabel()` | To print out the values readed
`.debugPrompt()` | To print out the steps for debugging purpose
`.check()`| To display an image that contain the reading side by side with the image of the target to do human checking.

## Steps taken
### 1.Find device

First, function below is used to find the physical device 

```python
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
```

This is just basic threshold and get contour function. Basically it will get the black block(which is conviniently the color of the meter)

The function above will return a greyscale image with flatten perspective as shown as below.

Input | Output
--- | ---
<img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/testing%20images/metertest2.png" width = "200"> | <img src = "https://github.com/Jawkx/opencv_meter_reader/blob/master/documentation%20pics/deviceimg.png" width = "200">

### 2.Find screen
After that, another threshold and contour function is implemented again. This time, lighter area was extracted instead. 

```python
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
```

Note that there will be a function `rotate()` being implemented. Because of unavoidable inconsistency of the picture. A backup rotate function is there to let the user rotate the screen orientation.

### 3.Extract digit image

This is one of the most complex part of the whole project. Due to the variance of lighting of different meter location and the minute difference between intensity of the screen and the text. The threshold value is always fluctuating. Thus a method was implemented where the program will loop through different value to find the optimal threshold base on the text/screen ratio

```python
    while True:
        ret,thresh = cv2.threshold(screen, threshold, 255, cv2.THRESH_BINARY_INV)
        x,y = thresh.shape
        whitecount = cv2.countNonZero(thresh)
        blackcount = (x*y) - whitecount
        ratio = (whitecount/(blackcount+whitecount))*100
        threshold = threshold - 17
        if ratio < ratiobase:
            break
```

Then to extract the text image.


