import numpy as np
import cv2
import time
import functions as fn
import os

### set the basePath ###
basePath = '/Users/sheelaahmed/Desktop/NAS/'

### define the trial number ###
trialNum = 1

### read in the reference image ###
ref_img = cv2.imread('time_mag_ref_img.png')  # ref image


### define all different colors to use in the rest of the program ###

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL

#######################################################################
### set the DETECTOR and MATCHER to whatever is desired for testing ###
#######################################################################
DETECTOR = 'SIFT'
MATCHER = 'FLANN'

PERSPECTIVE_TRANSFORM = False

### get the current time and date
timeStr = time.strftime("%Y-%b-%d_%Hh%Mm")
dateStr = time.strftime("%Y-%b-%d")

#### make a new directory to store each of the images ###
assistedCodeDir = ('Code_Results_{}-{}'.format(timeStr, trialNum))
os.makedirs(assistedCodeDir)


#trialPath = subjPath + trial

### filename stuff for when I move onto the videos ####
#videoFname = trialPath + 'world_out.mp4'  # converted video
#rawVideoFname = 'world.mp4'  # raw mp4 video.
#fixationCSVfName = trialPath + 'fixations.csv'  # sample csv file with Frame, X, Y of fixations

subj = 'mag_advertisements_{}'.format(trialNum)
subjPath = basePath + subj


### time how long it all takes ###
startTime = time.time()

### define the current file and total number of files in the subj path ###
currentFile = 1
totalFiles = len(os.listdir(subjPath))


while currentFile < totalFiles:

    test_img = cv2.imread('{}/mag_ad_{}.JPG'.format(subjPath, currentFile)) #test image
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    src_pts, dst_pts, M, mask = fn.feature_detect(test_img, ref_img, method=DETECTOR, matcher=MATCHER)

    detected_image = fn.object_detect(test_img, ref_img, dst_pts, mask)

    ### if you want to display the matches between both images ####
    if PERSPECTIVE_TRANSFORM == True:
        detected_image = fn.perspectiveTransform(test_img, ref_img, mask, M)

        cv2.namedWindow('Matches')
        cv2.imshow('Matches', detected_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    ### hstack both images
    displayImg = fn.hstack(test_img, detected_image, border=2)

    #####################################
    ### add text in the new window!!! ###
    #####################################

    ### create a display image name for each frame/image ###
    displayImg_name = ('displayImg_{}-'.format(currentFile))

    ### display each resulting window ###
    cv2.imshow(displayImg_name, displayImg)

    ### waitKey ###
    k = cv2.waitKey()  # display

    if (k & 0xff) == 27 or (k & 0xff) == 113 or (k & 0xff) == 81: #if Esc, q or Q is pushed
        k = 'exit' #exit program
        print('exiting...')
        sys.exit()
    #cv2.waitKey()

    ### write out the image into the new directory ###
    ### to change directory, go to 'assistedCodeDir' line at the top and change to desired directory name ###
    fname = '{}/{}.png'.format(assistedCodeDir, displayImg_name)
    cv2.imwrite(fname, displayImg)
    cv2.destroyAllWindows()

#    mag_ad += 1

    ### print out total time it took ###
elapsedTime = time.time() - startTime
print('Process is complete. Elapsed time is {}'.format(elapsedTime))