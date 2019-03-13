### 3/12/19
### testing the minDistToROI codes on fixation_dict, RoiDf, and nRowsImg and nColsImg

import pickle
import pandas as pd
import numpy as np
import cv2
# import logger


def findMinDistToROI(fixIn, roiDf, roiImagePath, numSegsX=4, numSegsY=8):
    # if fixIn['idx'] % 50 == 0:
        # logger.info('Finding minimum distance for fix ' + str(fixIn['idx']))

    def minDistToAnRoi(fixIn, roiIn, maskedImgIn):

        # In:  fixation point, RGB vals of an ROI, roiImage with ROI region defined by pixel RGB vals
        # Find minimum distance from a fixation point to a region of interest

        x = int(fixIn['fixNormX'] * np.shape(maskedImgIn)[1])
        y = int(fixIn['fixNormY'] * np.shape(maskedImgIn)[0])

        colorMask = cv2.inRange(maskedImgIn, roiIn['colorVal'], roiIn['colorVal'])

        distToPixelInRoi_px = [np.sqrt(np.nansum(np.power([x - mask_yx[1], y - mask_yx[0]], 2)))
                               for mask_yx in np.array(np.where(colorMask)).T]
        print(distToPixelInRoi_px)

        try:
            min = np.nanmin(distToPixelInRoi_px)
        except:
            min = np.nan
            pass
        # minDist to ROI of ALL ROI
        return min

    # Select only those RGB within the same segment as the fixation
    roiInSegDf = roiDf[(roiDf['Xseg'] == fixIn.loc['Xseg']) & (roiDf['Yseg'] == fixIn.loc['Yseg'])]
    # roiInSegDf = roiDf[(roiDf['Xseg']) == 2 & (roiDf['Yseg'] == 1)]
    print(roiInSegDf)
    # Check to see if there are ROI in this segment
    if (roiInSegDf.empty):
        # print('No roi in this fixation''s segment')
        return {'nearestROI': np.nan, 'distToNearestROI': np.nan, 'roiDistances': []}
    else:
        # roiImg = cv2.imread(roiImagePath)
        roiImg = roiImagePath

        # if( roiImg is  None):
            # logger.error('Invalid ROI file.')

        # Find pixels in the segment
        billHeightPx = np.shape(roiImg)[0] / numSegsX
        billWidthPx = np.shape(roiImg)[1] / numSegsY

        # np.min([billHeightPx,billWidthPx])
        # Mask the segment in the ROI image
        lBound = int(billWidthPx * fixIn['Xseg'])
        tBound = int(billHeightPx * fixIn['Yseg'])
        segMask = np.zeros(roiImg.shape[:2], np.uint8)
        segMask[tBound:int(tBound + billHeightPx), lBound:int(lBound + billWidthPx)] = 255
        roiSegImg = cv2.bitwise_and(roiImg, roiImg, mask=segMask)
        roiSegImg = cv2.cvtColor(roiSegImg, cv2.COLOR_BGR2RGB)



        # Fix not in an ROI.  Find nearest.

        # Find min fix-to-pixel dist for each ROI
        # minDistToROI_roi for this fix will be of len(roiInSegDf)
        minDistToROI_roi = roiInSegDf.apply(lambda roi: minDistToAnRoi(fixIn, roi, roiSegImg), axis=1)
        # Find min fix-to-roi distance among all distances
        minIdx = roiInSegDf['idx'].iloc[np.nanargmin(minDistToROI_roi)]
        minVal = np.min(minDistToROI_roi[np.isfinite(minDistToROI_roi)])  # nan min was having issues.

    minVal = minVal / np.min([billHeightPx, billWidthPx])

    #return {'nearestROI': minIdx, 'distToNearestROI': minVal, 'roiDistances': np.array(minDistToROI_roi.values,dtype=np.float) }
    return {'nearestROI': minIdx, 'distToNearestROI': minVal}


def createRoiDf(roiFileName, numSegsX=8, numSegsY=4):
    '''
    Input: Takes as input an image with color coded ROI in RGB space
    Returns: A dataframe where rows are regions of interest
    '''

    # logger.info('Finding ROI in image file.')

    # roiImg = cv2.imread(roiFileName)
    roiImg = roiFileName

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
