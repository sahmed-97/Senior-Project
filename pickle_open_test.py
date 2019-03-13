import pickle
import pandas as pd
# import cv2

print('Hello')
allTrialDataDf = pd.read_pickle('allData.pickle')
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


# print(roiDf)
print(allTrialDataDf)
# print(fixIn)
print('Done')
