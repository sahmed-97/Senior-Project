import numpy as np
import pandas as pd
import cv2
import pickle
import ROI_test as roi

roi_image = cv2.imread('roi400.png')

segX = 2
segY = 1
Frame = 41314
fixX = 430
fixY = 343

fix = []

fix.append({'Frame':Frame, 'Xseg': segX, 'Yseg': segY,
                        'fixX':fixX,'fixY':fixY})

fixIn = pd.DataFrame(fix)

# fixIn = pd.read_pickle('Subject_35_Fixations_2019-Mar-12_14h32m.pickle')
print("fixIn")
print(fixIn)
fixIn['fixNormX'] = fixIn['fixX'] / np.shape(roi_image)[1]
fixIn['fixNormY'] = fixIn['fixY'] / np.shape(roi_image)[0]

roiDf = roi.createRoiDf(roi_image, numSegsX=8, numSegsY=4)
print("roiDf")
print(roiDf)

dictOut = roi.findMinDistToROI(fixIn, roiDf, roi_image, numSegsX=4, numSegsY=8)
print("Dict Out")
print(dictOut)

print("Done!")
