# Code to read the image, convert the grayscale, find the contours and classify them using the trained CNN model


from keras.models import load_model
model = load_model('my_model.h5')
import numpy as np
import cv2
import imutils
import argparse
from skimage.filters import threshold_adaptive
import matplotlib.pyplot as plt
image = cv2.imread('image6.jpeg')
image = imutils.resize(image,width=320)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Rectangular kernel with size 5x5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
#apply blackhat and otsu thresholding
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,None)        
(_,cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])      
digits = []
boxes = []
for i,c in enumerate(cnts):
    if cv2.contourArea(c)<avgCntArea/10:
        continue
    mask = np.zeros(gray.shape,dtype="uint8")   #empty mask for each iteration

    (x,y,w,h) = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(mask,[hull],-1,255,-1)     #draw hull on mask
    mask = cv2.bitwise_and(thresh,thresh,mask=mask) #segment digit from thresh
    digit = mask[y-10:y+h+10,x-10:x+w+10]    #just for better approximation
    digit = cv2.resize(digit,(32,32))
    boxes.append((x,y,w,h))
    digits.append(digit)

digits = np.array(digits)
digits=digits/255.0
digits = digits.reshape(digits.shape[0],1,32,32)    #for Convolution Neural Networks
labels = model.predict_classes(digits)
cv2.imshow('Original',image)
cv2.imshow('Thresh',thresh)

for (x,y,w,h),label in sorted(zip(boxes,labels)):
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.putText(image,str(label),(x+10,y-4),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.imshow("Recognized",image)
    cv2.waitKey()

cv2.destroyAllWindows()





