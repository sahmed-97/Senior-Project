import pickle
import pandas as pd
import cv2
import numpy as np
# import cv2

# print('Hello')
# allTrialDataDf = pd.read_pickle('allData.pickle')
# roiDf = pd.read_pickle('roiCache.pickle')
# fixations = pd.read_csv('fixations.csv')
# fixations = pd.read_pickle('Subject_35_Fixations_2019-Mar-12_14h32m.pickle')

# segX = 2
# segY = 1
# Frame = 41314
# FixX = 430
# FixY = 343
#
# fix = []
#
# fix.append({'Frame':Frame, 'Xseg': segX, 'Yseg': segY,
#                         'FixX':FixX,'FixY':FixY})
#
# fixIn = pd.DataFrame(fix)
# fix['idx'] = range(0, len(fix))
def createRoiDf(roiImg, numSegsX, numSegsY):
    '''
    Input: Takes as input an image with color coded ROI in RGB space
    Returns: A dataframe where rows are regions of interest
    '''

    # logger.info('Finding ROI in image file.')

    # roiImg = cv2.imread(roiFileName)

    # Find pixels in the segment
    billHeightPx = np.shape(roiImg)[0] / numSegsY
    billWidthPx = np.shape(roiImg)[1] / numSegsX

    roiDicts = []

    for segX in range(numSegsX):
        for segY in range(numSegsY):

            # Mask the segment in the ROI image
            lBound = int(billWidthPx * segX)
            tBound = int(billHeightPx * segY)
            segMask = np.zeros(roiImg.shape[:2], np.uint8)
            segMask[tBound:int(tBound + billHeightPx), lBound:int(lBound + billWidthPx)] = 255
            roiSegImg = cv2.bitwise_and(roiImg, roiImg, mask=segMask)
            roiSegImg = cv2.cvtColor(roiSegImg, cv2.COLOR_BGR2RGB)

            # Find unique ROI within masked segment.
            # Surprisingly, this block takes the longest to compute.
            uniqueColors_roi = list(set(tuple(v) for m2d in roiSegImg for v in m2d))
            uniqueColors_roi.remove((255, 255, 255))
            uniqueColors_roi.remove((0, 0, 0))
            uniqueColors_roi = np.array(uniqueColors_roi, dtype="uint8")

            # Append ROI info to roiDicts (a list of dicts)
            for roiColor in uniqueColors_roi:
                # Find center of ROI on this segment
                roiMask = cv2.inRange(roiSegImg, roiColor, roiColor)
                maskCenter_YX = np.mean(np.where(roiMask), axis=1)

                roiDicts.append({'colorVal': np.array(roiColor, dtype="uint8"), 'Xseg': segX, 'Yseg': segY,
                                 'centroidX': maskCenter_YX[1],
                                 'centroidY': maskCenter_YX[0]})

    roiDf = pd.DataFrame(roiDicts)
    roiDf['idx'] = range(0, len(roiDf))

    return roiDf


if __name__ == "__main__":

    roi_image = cv2.imread('mag_roi.png')

    roiDf = createRoiDf(roi_image, numSegsX=5, numSegsY=3)

    roiDf.to_pickle("ROIDataframeMagazines_noGray.pickle")

# print(roiDf)
# print(allTrialDataDf)
# print(fixIn)
    print('Done')
