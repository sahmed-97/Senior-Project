from ID_assisted_defs import ID_, CTR_FRAME_, START_FRAME_, DUR_, X_, Y_  # Indices into fixationTable
from ID_assisted_defs import H_, W_  # fixBoxHW[]

import numpy as np
import cv2
import time
import functions as fn
import os
import pandas as pd

### set the basePath ###
basePath = '/Users/sheelaahmed/Desktop/NAS/'

### define subject number ###
subjNum = 35

### read in the reference image ###
ref_img = cv2.imread('time_mag_ref_img.png')  # ref image

### define width and height of the fixation box ###
fixBoxHW = (199,199)

### get the current time and date
timeStr = time.strftime("%Y-%b-%d_%Hh%Mm")
dateStr = time.strftime("%Y-%b-%d")

#### make a new directory to store each of the images ###
CodeDir = ('Video_Code_Results_{}-{}'.format(subjNum, timeStr))
os.makedirs(CodeDir)

# subj = 'mag_advertisements_{}'.format(trialNum)
trial = '00_00_000-43_54_973/'

### filename stuff for when I move onto the videos ####
subj = 'NAS_1_S{}'.format(subjNum)
subjPath = basePath + subj
trialPath = subjPath + trial

start_fixation = 369  # 330  # S28: Frames 6160 - 26350 = Fixations 300 - 3451
end_fixation = 3323  # to code all

exclude_fixations = []  # default if no fixations are excluded
exclude_fixations = list(np.r_[648:805, 1003:1711, 2290:2645])  # Don't code any fixations in these ranges


### define video and csv filenames ###
videoFname = trialPath+'world_out.mp4'  # converted video
rawVideoFname = 'world.mp4'  # raw mp4 video.
fixationCSVfName = trialPath+'fixations.csv'  # sample csv file with Frame, X, Y of fixations

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

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL

#######################################################################
### set the DETECTOR and MATCHER to whatever is desired for testing ###
#######################################################################
DETECTOR = 'SIFT'
MATCHER = 'FLANN'

PERSPECTIVE_TRANSFORM = False

VIDEO_WIDTH = 1280  # default - can be overwritten by raw video
VIDEO_HEIGHT = 720  # default

################################
### PUT SETUP_RUN STUFF HERE ###
################################


segmentedReferenceImage, segW, segH, nRowsRefImg, nColsRefImg = \
    fn.determine_segments_from_fname_and_image(refImgFname, refImg)

##############################################
#####comment all of these lines as well ######
##############################################

# Read data into a pandas dataframe
fixationDataPD = pd.read_csv(fixationCSVfName)

# Calculate the CENTER frame of each fixation and append it as a column to the dataFrame
ctr_fixation_series = pd.Series((fixationDataPD['start_frame_index'] + fixationDataPD['end_frame_index']) / 2,
                                name='center_frame')
fixationDataPD = pd.concat([fixationDataPD, ctr_fixation_series], axis=1)

FRAME_ = START_FRAME_
headers = list(fixationDataPD)

# Values are given in normalized coordinates: multiply by vidWidth * vidHeight, and invert the vertical dimension
fixationTable = np.transpose(np.array([fixationDataPD['id'], fixationDataPD['center_frame'], fixationDataPD['start_frame_index'],
                                       fixationDataPD['duration'], fixationDataPD['norm_pos_x'] * VIDEO_WIDTH,
                                       VIDEO_HEIGHT - (fixationDataPD['norm_pos_y'] * VIDEO_HEIGHT)]))
fixationTable = fixationTable.astype(int)  # Convert center frames to integer
# fixationTable[:, 0] = fixationTable[:,0].astype(int)  # Convert center frames to integer

### get number of fixations in the table
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
fixTableIdx = start_fixation - 1  # Index into table is fixation# - 1

nextFixationFrame = fixationTable[fixTableIdx, FRAME_]

firstFixationFrame = nextFixationFrame

firstPass = True

### time how long it all takes ###
startTime = time.time()

### define the current file and total number of files in the subj path ###
#currentFile = 1
#totalFiles = len(os.listdir(subjPath))

currentFrame = firstFixationFrame
img_index = 0

while currentFrame < min(NFRAMES, vidObjDict['nFrames'] - 1):

# while currentFile < totalFiles:

    if currentFrame < firstFixationFrame:
        fn.skip_forward_to_first_desired_frame(vidObjDict['vidObj'], firstFixationFrame, currentFrame)
    # grab the current frame
    frame, vidObjDict = fn.grab_video_frame(avObj, streamObj, vidObjDict, ticksPerFrame, frameNumToRead=nextFixationFrame)

    currentFrame = vidObjDict['currentFrame']

    fixPosXY = [0.0, 0.0]
    fixWinUL = [0.0, 0.0]

    if currentFrame == nextFixationFrame:

        print('Current frame = {}'.format(currentFrame))

        ### get fixation information from csv file ###
        # fixPosXY, fixWinUL, fixDur, fixID, inFrame, fixationInfo = fn.fetch_fixation_data_from_fixation_table(fixationTable, fixTableIdx, fixBoxHW, vidObjDict, currentFrame)

        ##################################################################
        ### do the feature detection based on the fixation coordinates ###
        ##################################################################

        src_pts, dst_pts, M, mask = fn.feature_detect(currentFrame, ref_img, method=DETECTOR, matcher=MATCHER)

        ### edit this as well to match parameters and outputs ###
        index_x, index_y, obj_height, obj_width = fn.object_detect(ref_img, dst_pts, mask)

        ### if you want to display the matches between both images ####
        if PERSPECTIVE_TRANSFORM == True:
            detected_image = fn.perspectiveTransform(currentFrame, ref_img, mask, M)

            cv2.namedWindow('Matches')
            cv2.imshow('Matches', detected_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        ### hstack both images
        # displayImg = fn.hstack(test_img, detected_image, border=2)

        #####################################
        ### add text in the new window!!! ###
        #####################################

        ### create a display image name for each frame/image ###
        displayImg_name = ('displayImg_{}-'.format(currentFrame))

        ### display each resulting window ###
        displayImg = fn.display_image(ref_img, frame, index_x, index_y, obj_height, obj_width)
        # cv2.imshow(displayImg_name, displayImg)

        ### write out the image into the new directory ###
        ### to change directory, go to 'assistedCodeDir' line at the top and change to desired directory name ###
        fname = '{}/{}.png'.format(CodeDir, displayImg_name)
        cv2.imwrite(fname, displayImg)
        img_index += 1

        # If the next fixation is in the list of exclusions, skip through them
        while fixTableIdx in exclude_fixations:
            print("fixTableIdx {} is in the list of excluded fixations; skipping it ...".format(fixTableIdx))
            fixTableIdx = fixTableIdx + 1   # prepare for next fixation in table

        # Check for end of trial
        if fixTableIdx < min(end_fixation, (nFixations - 1)):
            nextFixationFrame = fixationTable[fixTableIdx, FRAME_]
        else:
            break

    ###### if currentFrame != nextFixationFrame, don't process it ######
    else:
        ### Note: There are occasionally errors in the fixation files where adjacent fixations have the same frame number. ###
        ### This leads to loops where we just keep seeking higher frames, since we are already past the frame we wanted. ###
        ### So check for the special case where currentFrame > nextFixation frame ###

        ### Error in fixation.csv file - skip this fixation and go to the next one ###
        if currentFrame > nextFixationFrame:

            ### prepare for next fixation in table ###
            fixTableIdx = fixTableIdx + 1
            if fixTableIdx < min(end_fixation, (nFixations - 1)):
                nextFixationFrame = fixationTable[fixTableIdx, FRAME_]
            else:
                break

cv2.destroyAllWindows()
### print out total time it took ###
elapsedTime = time.time() - startTime
print('Process is complete. Elapsed time is {}'.format(elapsedTime))
