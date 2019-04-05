from ID_assisted_defs import ID_, CTR_FRAME_, START_FRAME_, DUR_, X_, Y_  # Indices into fixationTable
from ID_assisted_defs import H_, W_  # fixBoxHW[]

import numpy as np
import cv2
import time
import functions as fn
import os
import xlrd
import pandas as pd
import av
import matplotlib.pyplot as plt
import pickle


### set the basePath ###
basePath = '/Users/sheelaahmed/Desktop/NAS/'

refImgFname = basePath + 'Ref_Img_PNG_378x500.png'

### read in ROI image, ROI Labels pickle file, and DF pickle file ###
roiImagePath = basePath + 'mag_roi_gray.png'
roiImage = cv2.imread(roiImagePath)

# roiLabelsPath = basePath + 'roiLabels.xlsx'
# roiLabels = pd.read_excel(roiLabelsPath)

roiDfPath = basePath + 'ROIDataframeMagazines.pickle'
roiDf = pd.read_pickle(roiDfPath)

### DEFINE SUBJECT NUMBER ###
subjNum = 2

### read in the reference image ###
ref_Img = cv2.imread(refImgFname)  # ref image

### define width and height of the fixation box ###
fixBoxHW = (199,199)

### get the current time and date
timeStr = time.strftime("%Y-%b-%d_%Hh%Mm")
dateStr = time.strftime("%Y-%b-%d")

#### make a new directory to store each of the images ###
CodeDir = ('Subject {}_Results_{}'.format(subjNum, timeStr))
os.makedirs(CodeDir)

# subj = 'mag_advertisements_{}'.format(trialNum)
trial = '/trial2/'
fixations = '/exports/000/'

### filename stuff for when I move onto the videos ####
# subj = 'NAS_1_S{}/'.format(subjNum)
subj = 'magazine_testing/'
subjPath = basePath + subj
trialPath = subjPath + trial + fixations

start_fixation = 80  # 330  # S28: Frames 6160 - 26350 = Fixations 300 - 3451
end_fixation = 85 # to code all

### list of fixations to exclude(calibration etc) ###
# exclude_fixations = []
exclude_fixations = list(np.r_[130:136])

### define video and csv filenames ###
# videoFname = basePath + 'vlc-record-2019-02-18-10h47m24s-world.mp4' ### new cropped video file for testing! ###
rawVideoFname = 'world.mp4'  # raw mp4 video.
fixationCSVfName = trialPath+'fixations.csv'  # sample csv file with Frame, X, Y of fixations

########### set up filenames for new csv files to be exported ###########
pickle_object_filename = 'Subject_{}_Fixations_{}.pickle'.format(subjNum, timeStr)
csv_ROI_filename = 'Subject_{}_ROI_Results_{}.csv'.format(subjNum, timeStr)

### define all different colors to use in the rest of the program ###
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

NFRAMES = 10000000  # 9999

FONT = cv2.FONT_HERSHEY_SIMPLEX

### set a boolean for if the csv file to be exported in the end has already been opened and created ###
csvFileOpened = False

#######################################################################
### set the DETECTOR and MATCHER to whatever is desired for testing ###
#######################################################################
DETECTOR = 'SIFT'
MATCHER = 'FLANN'

VIDEO_WIDTH = 1280  # default - can be overwritten by raw video
VIDEO_HEIGHT = 720  # default

### determine segments in the reference image ###
segmentedReferenceImage, segW, segH, nRowsRefImg, nColsRefImg = \
    fn.determine_segments_from_fname_and_image(refImgFname, ref_Img)

# Read data into a pandas dataframe
fixationDataPD = pd.read_csv(fixationCSVfName)

# Calculate the CENTER frame of each fixation and append it as a column to the dataFrame
ctr_fixation_series = pd.Series((fixationDataPD['start_frame_index'] + fixationDataPD['end_frame_index']) / 2,
                                name='center_frame')
fixationDataPD = pd.concat([fixationDataPD, ctr_fixation_series], axis=1)

### define start frame from the imported info of ID_assisted_defs ###
FRAME_ = START_FRAME_
headers = list(fixationDataPD)

# Values are given in normalized coordinates: multiply by vidWidth * vidHeight, and invert the vertical dimension
fixationTable = np.transpose(np.array([fixationDataPD['id'], fixationDataPD['center_frame'], fixationDataPD['start_frame_index'],
                                       fixationDataPD['duration'], fixationDataPD['norm_pos_x'] * VIDEO_WIDTH,
                                       VIDEO_HEIGHT - (fixationDataPD['norm_pos_y'] * VIDEO_HEIGHT)]))
fixationTable = fixationTable.astype(int)  # Convert center frames to integer
# fixationTable[:, 0] = fixationTable[:,0].astype(int)  # Convert center frames to integer

### get number of fixations in the table
# nFixations = end_fixation - start_fixation
nFixations = len(fixationTable)

fixationsToCode = nFixations - len(exclude_fixations)  # The total number of fixations that need to be coded
print("nFixations = {} - excluded fixations {} = {} fixations to code".format(nFixations, len(exclude_fixations), fixationsToCode))


### get dictionary of all elements in raw video ###
# vidObjDict, avObj, streamObj, ticksPerFrame = fn.open_raw_mp4(subjPath + rawVideoFname)
vidObjDict, avObj, streamObj, ticksPerFrame = fn.open_raw_mp4(subjPath + trial + rawVideoFname)
vidObjDict['fName'] = rawVideoFname
videoWH = vidObjDict['width'], vidObjDict['height']

#### startatFix is in the setup_run function ###
#### basically move all important stuff from that function into this code ###
#### only focus on one subject at a time ###

### Index into table is fixation# - 1 ##
fixTableIdx = start_fixation - 1

### define next fixation from fixation table ###
nextFixationFrame = fixationTable[fixTableIdx, FRAME_]

firstPass = True

### time how long it all takes ###
startTime = time.time()

### index the current frame to iterate through the frames in table ###
currentFrame = 0

### initialize array for fixations in reference image ###
reference_frame_fixations = []


################################################################
### create headers and strings for csv files to be exported ###
csv_object_header = ' Xseg,Yseg,Frame,Fixation,FrameFixX,' \
                'FrameFixY,RefFixX,RefFixY, nearestROI'
output_string_object = []

# fixation_dict_header = 'Xseg,Yseg,norm_pos_x,norm_pos_y'

### initialize dictionary to fill with fixations and segments ###
###to be used in distToROI at end ######
fixation_dict = {}

####################################
### begin looping through frames ###
####################################
while currentFrame < min(NFRAMES, vidObjDict['nFrames'] - 1):

    ### make a copy of the reference image ###
    ref_img = ref_Img.copy()

    ###define current frame as it iterates through the table ###
    currentFrame = nextFixationFrame
    # firstFixationFrame = nextFixationFrame

    # if currentFrame < firstFixationFrame:
    # fn.skip_forward_to_first_desired_frame(vidObjDict['vidObj'], firstFixationFrame, currentFrame)
    # grab the current frame
    frame, vidObjDict = fn.grab_video_frame(avObj, streamObj, vidObjDict, ticksPerFrame, frameNumToRead=nextFixationFrame)

    currentFrame = vidObjDict['currentFrame']

    fixPosXY = [0.0, 0.0]
    fixWinUL = [0.0, 0.0]

    if currentFrame == nextFixationFrame:

        print('Current frame = {}'.format(currentFrame))

        ### get fixation information from csv file ###
        fixPosXY, fixWinUL, fixDur, fixID, inFrame, fixationInfo = fn.fetch_fixation_data_from_fixation_table(fixationTable, fixTableIdx, fixBoxHW, vidObjDict, currentFrame)

        fixRegionImg = frame[fixWinUL[0]:fixWinUL[0] + fixBoxHW[H_], fixWinUL[1]:fixWinUL[1] + fixBoxHW[W_]].copy()

        ### Make a mask: 0s everywhere except the window surrounding fixation (1s) ###
        frameMask = np.zeros(frame.shape, np.uint8)
        frameMask[fixWinUL[0]:fixWinUL[0] + fixBoxHW[H_], fixWinUL[1]:fixWinUL[1] + fixBoxHW[W_]] = 1
        frameMask = cv2.cvtColor(frameMask, cv2.COLOR_BGR2GRAY)

        ##################################################################
        ### do feature detection based on the fixation coordinates.    ###
        ### need to use try/except in order to work around any error.  ###
        ##################################################################
        try:
            distance_points, Matrix, mask, good_matches, frame_kp, ref_kp = fn.feature_detect(frame, frameMask, ref_img, method=DETECTOR, matcher=MATCHER)
        except Exception as errMsg:
            print("cannot get src and dst pts for frame {} to form homography ... {}".format(currentFrame, errMsg))
            currentFrame += 1

        ##################################################################
        ######## same with object detection --> try/except      ##########
        ##################################################################
        try:
            index_x, index_y, obj_height, obj_width, x_avg, y_avg = fn.object_detect(ref_img, distance_points, segW, segH, nRowsRefImg, nColsRefImg, mask)
        except Exception as errMsg:
            print("Error has occured for frame {} ... {}".format(currentFrame, errMsg))
            index_x = 0
            index_y = 0
            obj_width = 1
            obj_height = 1
            currentFrame += 1

        ### define the x and y positions in the reference image ###
        ref_pos_x = int(index_x / obj_width)
        ref_pos_y = int(index_y / obj_height)

        ### create a display image name for each frame/image ###
        displayImg_name = ('displayImg_{}-'.format(currentFrame))


        ### display each resulting window ###
        try:
            ref_fix, displayImg = fn.object_display_image(ref_img, frame, fixRegionImg, fixPosXY, index_x, index_y, obj_height, obj_width, currentFrame, fixTableIdx, Matrix, mask, good_matches, frame_kp, ref_kp)
            display_img = True
        except Exception as errMsg:
            display_img = False
            ref_fix = (0,0)
            currentFrame += 1
            pass
        # cv2.imshow(displayImg_name, displayImg)

        ### append coordinate point to the list of reference coordinates ###
        reference_frame_fixations.append([ref_fix[0], ref_fix[1]])

        ### write out the image into the new directory ###
        ### to change directory, go to 'assistedCodeDir' line at the top and change to desired directory name ###
        fname = '{}/{}.jpg'.format(CodeDir, displayImg_name)
        if display_img == True:
            cv2.imwrite(fname, displayImg)
        else:
            currentFrame += 1
            pass

        # If the next fixation is in the list of exclusions, skip through them
        while fixTableIdx in exclude_fixations:
            print("fixTableIdx {} is in the list of excluded fixations; skipping it ...".format(fixTableIdx))
            fixTableIdx = fixTableIdx + 1    # prepare for next fixation in table

        # Check for end of trial
        if fixTableIdx < min(end_fixation, (nFixations - 1)):
            nextFixationFrame = fixationTable[fixTableIdx, FRAME_]
        else:
            break

        ### create output string for .csv file that will be exported ###
        # output_string_list = "{}, {}, {}, {}, {}, {}, {}, {}".format(ref_pos_x, ref_pos_y, currentFrame, fixTableIdx, fixPosXY[0],
        #                                                 fixPosXY[1], ref_fix[0], ref_fix[1])

        output_string_list = {'Xseg':ref_pos_x,'Yseg':ref_pos_y,'Frame':currentFrame,'Fixation':fixTableIdx,'fixX':fixPosXY[0]
                                 ,'fixY':fixPosXY[1],'RefFixX':ref_fix[0],'RefFixY':ref_fix[1]}
        output_string_object.append(output_string_list)


    ###### if currentFrame != nextFixationFrame, don't process it ######
    else:
        ### Note: There are occasionally errors in the fixation files where adjacent fixations have the same frame number. ###
        ### This leads to loops where we just keep seeking higher frames, since we are already past the frame we wanted. ###
        ### So check for the special case where currentFrame > nextFixation frame ###

        if currentFrame > nextFixationFrame:

            ### prepare for next fixation in table ###
            fixTableIdx = fixTableIdx + 1
            if fixTableIdx < min(end_fixation, (nFixations - 1)):
                nextFixationFrame = fixationTable[fixTableIdx, FRAME_]
            else:
                break

    fixTableIdx = fixTableIdx + 1

# print('fixation dict')
# print(fixation_dict)

fixation_dict = pd.DataFrame(output_string_object)
# fixation_dict.drop([fixation_dict['refFixX'] == -1].index)
# fixation_dict.drop([fixation_dict['refFixY'] == -1].index)
#
# print('fixation dict')
# print(fixation_dict)
# print('ROI Df')
# print(roiDf)

##########################################################################################
####### use new fixations in reference image to begin calculating distance to ROI ########
####### as well as other visualizations i.e. heat maps, plots of fixations, etc... #######
##########################################################################################

# # Normalize fixation locations within the image
fixation_dict['fixNormX'] = fixation_dict['fixX'] / np.shape(ref_Img)[1]
fixation_dict['fixNormY'] = fixation_dict['fixY'] / np.shape(ref_Img)[0]


# distToROI_fix = fixation_dict.apply(lambda row: fn.find_min_dist_to_ROI(row,roiDf,fixationDataPD, roiImage, nRowsRefImg, nColsRefImg),axis=1)
# print(distToROI_fix)

# Add the values implied by dict keys in distToROI_fix to fixDf
# fixation_dict = fixation_dict.combine_first(pd.DataFrame.from_records(distToROI_fix))
# fixation_dict['subjectID'] = subID
# print('ROI matches')
# print(distToROI_fix)

print('fixation dictionary')
print(fixation_dict)
#
nearestROIVals = fixation_dict['nearestROI'] #### NEED TO EDIT STILL ###

pd.DataFrame(fixation_dict).hist(column='nearestROI')
# fixation_dict.to_pickle(pickle_object_filename)

################################################################################
####### PLOTTING HEAT MAPS ###############

graspHeatmapRes_yx = [239,200]
scaleFactor = np.shape(ref_Img)[1] / graspHeatmapRes_yx[1]

binPixX = np.shape(ref_Img)[1] / graspHeatmapRes_yx[1]
binPixY = np.shape(ref_Img)[0] / graspHeatmapRes_yx[0]
gridOffset = (np.shape(ref_Img)[0] / graspHeatmapRes_yx[0])/2

subHeatMap_xy = fn.makeHeatMap(ref_Img, fixation_dict, nColsRefImg, nRowsRefImg, fixationDataPD, binPixX, binPixY, gridOffset, withDuration=False)

# gaussStdPx = np.int(scaleFactor * 2)
# print(gaussStdPx)
gaussStdPx = 129

if np.mod(gaussStdPx,2) == 0: #has to be odd
    gaussStdPx = gaussStdPx+1

subHeatMap_xy = np.array(subHeatMap_xy,dtype=np.uint8)

imStack_xy = fn.normalizeHeatMapWithinBillFace(ref_Img, subHeatMap_xy, numSegsX=5,numSegsY=3,
                                    gaussStdPx = gaussStdPx, colormap=cv2.COLORMAP_HOT,# )  # COLORMAP_HOT   COLORMAP_JET
                                    heatmapAlpha=0.6,
                                   overlay=True)

fig, ax = plt.subplots(figsize=(8, 8), dpi= 300, facecolor='w', edgecolor='k')
ax.imshow(imStack_xy)
# ax.imshow(imStack_xy[:np.shape(imStack_xy)[0]/2,:,:])
ax.axis('off')

################################################################################
### plot reference fixations onto reference image ###
figure, axis = plt.subplots(figsize = (15,15), dpi = 80, facecolor = 'w', edgecolor = 'k')
axis.imshow(ref_img)
plt.xlim([0, np.shape(ref_img)[1]])
plt.ylim([np.shape(ref_img)[0], 0])
for x, y in reference_frame_fixations:
    plt.scatter(x, y, s = 20, c = 'r')
plt.show()
plt.savefig(basePath + 'Subject_{}_Fixation_Plot_{}.png'.format(subjNum, timeStr), bbox_inches='tight', transparent=True)
cv2.destroyAllWindows()


### print out total time it took ###
elapsedTime = time.time() - startTime
print('Subject {} process is complete. Elapsed time is {} for {} fixations'.format(subjNum, elapsedTime, end_fixation - start_fixation))