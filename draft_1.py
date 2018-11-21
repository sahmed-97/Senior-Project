# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:22:32 2018

@author: sheel
"""

import cv2
import numpy as np
import pandas as pd
import av
import time
import math
import os
import csv
import sys


#%% convert all color images/frames to grayscale

def img_to_grayscale(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return gray

#%% Detect features between images through SIFT, ORB, SURF
def feature_detection(img1, img2, detection_method = 'ORB'):
    
    
    if detection_method = 'ORB':
        ##ORB detector

        sd = cv2.ORB_create()


    elif detection_method = 'SIFT':
        
        sd = cv2.xfeatures2d.SIFT_create()

    elif detection_method = 'SURF':
        
        sd = cv2.SURF(400)
        
        
    kp1, des1 = sd.detectAndCompute(img1, None)
    kp2, des2 = sd.detectAndCompute(img2, None)
    
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
        
    detected_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

    return detected_img

#%%reading the video file
    
def read_video(raw_vid):
    
    avObj = None  # initialize to null value

    avObj = av.open(rawVidFname)
    # avObj2 = cv2.videoCapture(rawVidFname)
    streamObj = next(s for s in avObj.streams if s.type == 'video')

    nFrames = streamObj.frames
    print(nFrames)
    ticksPerFrame = streamObj.rate / streamObj.time_base
    width = streamObj.width


    vidObjDict = {'avObj':avObj, 'streamObj':streamObj,
                  'nFrames':nFrames, 'ticksPerFrame':ticksPerFrame, 'type':'raw',
                  'width':streamObj.width, 'height':streamObj.height}
    
    return vidObjDict

#%% putting it all together
    
if __name__ == "__main__":
    
    
    videoFname = 'C:/Users/sheel/Desktop/4th Year/Senior Project/eye1.mp4'
    read_video(videoFname)
    
    
    img1 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame0.jpg')
    img2 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame2594.jpg')
    
    
    
    

