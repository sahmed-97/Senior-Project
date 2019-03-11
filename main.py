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

refImgFname = basePath + 'RefImg_Series_205x490.jpg'

### read in ROI image, ROI Labels pickle file, and DF pickle file ###
roiImagePath = basePath + 'roi400.png'
roiImage = cv2.imread(roiImagePath)

roiLabelsPath = basePath + 'roiLabels.xlsx'
roiLabels = pd.read_excel(roiLabelsPath)

roiDfPath = basePath + 'roiCache.pickle'
roiDf = pd.read_pickle(roiDfPath)

### DEFINE SUBJECT NUMBER ###
subjNum = 35

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
trial = '00_00_000-43_54_973/'

### filename stuff for when I move onto the videos ####
subj = 'NAS_1_S{}/'.format(subjNum)
subjPath = basePath + subj
trialPath = subjPath + trial

start_fixation = 3850  # 330  # S28: Frames 6160 - 26350 = Fixations 300 - 3451
end_fixation = 3865  # to code all

allTrialDataDf = pd.read_pickle(basePath + 'allData.pickle')
fixIn = allTrialDataDf.iloc[start_fixation:end_fixation]

### list of fixations to exclude(calibration etc) ###
exclude_fixations = list(np.r_[3920:4115, 4120:4315, 4320:5000, 5338:5652])

### define video and csv filenames ###
# videoFname = basePath + 'vlc-record-2019-02-18-10h47m24s-world.mp4' ### new cropped video file for testing! ###
rawVideoFname = 'world.mp4'  # raw mp4 video.
fixationCSVfName = trialPath+'fixations.csv'  # sample csv file with Frame, X, Y of fixations

########### set up filenames for new csv files to be exported ###########
csv_object_filename = 'Subject_{}_Fixations_{}.csv'.format(subjNum, timeStr)
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
vidObjDict, avObj, streamObj, ticksPerFrame = fn.open_raw_mp4(subjPath + rawVideoFname)
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

### initialize dictionary to fill with fixations and segments ###
###to be used in distToROI at end ######
# fixation_dict = {}

#############################################
### create headers and strings for csv files to be exported ###
csv_object_header = 'Frame,Fixation,FrameFixX,' \
                'FrameFixY,RefFixX,RefFixY,Column,Row'
output_string_object = []

csv_ROI_header = 'FixNum,refFixX,refFixY,nearestROI,distToNearestROI'
output_string_ROI = []

#############################################

### begin looping through frames ###
while currentFrame < min(NFRAMES, vidObjDict['nFrames'] - 1):

    ### make a copy of the reference image ###
    ref_img = ref_Img.copy()

    ###define current frame as it iterates through the table ###
    currentFrame = nextFixationFrame
    # firstFixationFrame = nextFixationFrame
# while currentFile < totalFiles:

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
            currentFrame += 1

        ### define the x and y positions in the reference image ###
        ref_pos_x = int(index_x / obj_width)
        ref_pos_y = int(index_y / obj_height)

        ### if you want to display the matches between both images ####
        if PERSPECTIVE_TRANSFORM == True:
            detected_image = fn.perspectiveTransform(currentFrame, ref_img, mask, Matrix)

            cv2.namedWindow('Matches')
            cv2.imshow('Matches', detected_image)
            cv2.waitKey()
            cv2.destroyAllWindows()


        ### create a display image name for each frame/image ###
        displayImg_name = ('displayImg_{}-'.format(currentFrame))


        ### display each resulting window ###
        ref_fix, displayImg = fn.object_display_image(ref_img, frame, fixRegionImg, fixPosXY, index_x, index_y, obj_height, obj_width, currentFrame, fixTableIdx, Matrix, mask, good_matches, frame_kp, ref_kp)
        # cv2.imshow(displayImg_name, displayImg)

        ### append coordinate point to the list of reference coordinates ###
        reference_frame_fixations.append([ref_fix[0], ref_fix[1]])

        ### write out the image into the new directory ###
        ### to change directory, go to 'assistedCodeDir' line at the top and change to desired directory name ###
        fname = '{}/{}.jpg'.format(CodeDir, displayImg_name)
        cv2.imwrite(fname, displayImg)

        # If the next fixation is in the list of exclusions, skip through them
        while fixTableIdx in exclude_fixations:
            print("fixTableIdx {} is in the list of excluded fixations; skipping it ...".format(fixTableIdx))
            fixTableIdx = fixTableIdx + 1    # prepare for next fixation in table

        # Check for end of trial
        if fixTableIdx < min(end_fixation, (nFixations - 1)):
            nextFixationFrame = fixationTable[fixTableIdx, FRAME_]
        else:
            break

        ####### if lines below dont work, try this method with looping at end of code with plt ########
        fixation_dict = {'norm_pos_x':fixationDataPD['norm_pos_x'], 'norm_pos_x':fixationDataPD['norm_pos_y'], 'Xseg':ref_pos_x, 'Yseg':ref_pos_y}

            ###### test out the norm_pos_x and norm_pos_y columns to match dnfsdkk#####
        ################################################################################
        dictOut = fn.find_min_dist_to_ROI(fixation_dict, roiImage, nRowsRefImg, nColsRefImg)

        output_ROI_string_list = '{}, {}, {}, {}, {}'.format(fixTableIdx, ref_fix[0], ref_fix[1], dictOut['nearestROI'], dictOut['distToNearestROI'])
        output_string_ROI.append(output_ROI_string_list)

        if not csvFileOpened:  # If we haven't started writing to the failsafe file yet:
            fn.write_stringlist_to_csv_file(basePath + csv_ROI_filename, csv_ROI_header, output_string_ROI)
            csvFileOpened = True

        else:  # it's already started; just append the most recent row
            fn.append_stringlist_to_csv_file(basePath + csv_ROI_filename, output_ROI_string_list)
        ################################################################################

        ### create output string for .csv file that will be exported ###
        output_string_list = "{}, {}, {}, {}, {}, {}, {}, {}".format(currentFrame, fixTableIdx, fixPosXY[0],
                                                        fixPosXY[1], ref_fix[0], ref_fix[1], ref_pos_x, ref_pos_y)

        output_string_object.append(output_string_list)

        ######### append data from object detection to another csv file ##########
        if not csvFileOpened:  # If we haven't started writing to the failsafe file yet:
            fn.write_stringlist_to_csv_file(basePath + csv_object_filename, csv_object_header, output_string_object)
            csvFileOpened = True

        else:  # it's already started; just append the most recent row
            fn.append_stringlist_to_csv_file(basePath + csv_object_filename, output_string_list)

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

# print(reference_frame_fixations)

# x, y = reference_frame_fixations

##########################################################################################
####### use new fixations in reference image to begin calculating distance to ROI ########
####### as well as other visualizations i.e. heat maps, plots of fixations, etc... #######
##########################################################################################

# fixDf = pd.read_csv(basePath + csv_object_filename)

figure, axis = plt.subplots(figsize = (15,15), dpi = 80, facecolor = 'w', edgecolor = 'k')
axis.imshow(ref_img)
plt.xlim([0, np.shape(ref_img)[1]])
plt.ylim([np.shape(ref_img)[0], 0])
for x, y in reference_frame_fixations:
    # dictOut = fn.find_min_dist_to_ROI(fixIn, roiImage, nRowsRefImg, nColsRefImg)
    #
    # output_ROI_string_list = '{}, {}, {}, {}, {}'.format(fixTableIdx, ref_fix[0], ref_fix[1], dictOut['nearestROI'], dictOut['distToNearestROI'])
    # output_string_ROI.append(output_ROI_string_list)
    #
    # if not csvFileOpened:  # If we haven't started writing to the failsafe file yet:
    #     fn.write_stringlist_to_csv_file(basePath + csv_ROI_filename, csv_ROI_header, output_string_ROI)
    #     csvFileOpened = True
    #
    # else:  # it's already started; just append the most recent row
    #     fn.append_stringlist_to_csv_file(basePath + csv_ROI_filename, output_ROI_string_list)

    plt.scatter(x, y, s = 20, c = 'r')
plt.show()
plt.savefig(basePath + 'Subject {} Fixation Plot {}.png'.format(subjNum, timeStr), dpi = 'figure')
cv2.destroyAllWindows()


### print out total time it took ###
elapsedTime = time.time() - startTime
print('Subject {} process is complete. Elapsed time is {} for {} fixations'.format(subjNum, elapsedTime, end_fixation - start_fixation))
