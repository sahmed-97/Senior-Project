# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Read in and play a video

import numpy as np
import cv2

cap = cv2.VideoCapture('C:/Users/sheel/Desktop/4th Year/Senior Project/eye1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#%% extract single frame?

###Need to fix up###

import numpy
import cv2


#pass in video file
vid = cv2.VideoCapture('C:/Users/sheel/Desktop/4th Year/Senior Project/eye1.mp4')

count = 1
success = 1


while success:
    while count < 5:
        success, image = vid.read()
        cv2.imwrite('frame%d.jpg' % count, image)
        count += 1
        
    

#%% use PyAV to read and play video



#%% Match two images
        
import cv2
import numpy as np
#import matplotlib.pyplot as plt


#449img1 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/Original_panda.png')
#img2 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/Window_test2.tif')

img1 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame0.jpg')
img2 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame2594.jpg')

##ORB detector
#Sift not working properly 
orb = cv2.ORB_create()

##find kypoints and descriptors thru ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

##BFMatcher with default parameters
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
 
        
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

#cv2.imshow('Image 1', img1)
#cv2.imshow('Image 2', img2)
cv2.imshow('Matching results', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%for loop to loop through images


#%% build feature detection into a function

#%%
