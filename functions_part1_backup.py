import cv2
import numpy as np
import re
import time
import sys
import av
import matplotlib.pyplot as plt

from ID_assisted_defs import ID_, CTR_FRAME_, START_FRAME_, DUR_, X_, Y_  # Indices into fixationTable
from ID_assisted_defs import H_, W_  # fixBoxHW[]

DETECTOR = "SIFT"

MATCHER = "FLANN"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL


####################################################################################################
####################################################################################################
########## can determine the number of segments in the reference image based on the filename #######

def determine_segments_from_fname_and_image(fname, img):
    # Look for grid size in filename using regular expressions
    sizePattern = re.compile('\d+')  # Regular expression: look for multi-character integers
    segmentWHList = sizePattern.findall(fname)  # Returns list of integers found in filename

    if len(segmentWHList) == 2:  # There are two integers; assume that they are segW and segH
        segW = int(segmentWHList[0])
        segH = int(segmentWHList[1])
        imgIsSegmented = True

        # How many rows and columns of items are there in the reference image?
        refImgHeight, refImgWidth  = img.shape[0:2]

        nRowsImg = refImgHeight // segH
    nColsImg = refImgWidth // segW
    print("Based on filename '{}', assume reference image is made up of {} x {} regions"
          .format(fname, nRowsImg, nColsImg))

    return imgIsSegmented, segW, segH, nRowsImg, nColsImg

######################################################################################################
######################################################################################################
######### this function will be for first getting fixation info from the fixations.csv file ##########

def fetch_fixation_data_from_fixation_table(fixationTable, fixTableIdx, fixBoxHW, vidObjDict,
                                            currentFrame):
    ### this code will get fixation information from the fixations.csv file that is exported after running thru PupilPlayer ###

    ### define column of coordinate points from the fixation table ###
    fixPosXY = fixationTable[fixTableIdx,X_], fixationTable[fixTableIdx,Y_]

    ### Note order: U is Y, L is X ###
    fixWinUL = [fixPosXY[1] - int(fixBoxHW[H_]/2), fixPosXY[0] - int(fixBoxHW[W_]/2)]

    ### get duration of fixations (fixDur) and ??? (fixID) ###
    fixDur = fixationTable[fixTableIdx, DUR_]
    fixID = fixationTable[fixTableIdx, ID_]

    ### if the fixation is not in the frame of the video, set a boolean to inFrame ###
    if (fixWinUL[1] < 0) or (fixWinUL[1] > (vidObjDict['width'] - fixBoxHW[W_])) or \
            (fixWinUL[0] < 0) or (fixWinUL[0] > (vidObjDict['height'] - fixBoxHW[H_])):  # not in frame
        inFrame = False
    else:
        inFrame = True

    ### return dictionary of all the elements from fixations.csv ###
    fixationInfo = {"fixPosXY":fixPosXY, "fixWinUL":fixWinUL, "fixDur":fixDur,
                    "currentFrame":currentFrame, "fixID":fixID, "inFrame":inFrame,
                    'matchRatio':None, 'topMatch':None, 'secondMatch':None, }

    return fixPosXY, fixWinUL, fixDur, fixID, inFrame, fixationInfo

######################################################################################################
######################################################################################################
##### this function will be for using PyAV to open the raw video file and extracting info/frames #####

def open_raw_mp4(rawVidFname):

    ### initialize to null value ###
    avObj = None

    ### open the video using PyAV ###
    avObj = av.open(rawVidFname)

    ### define the stream object that extracts the individual frames in the video ###
    streamObj = next(s for s in avObj.streams if s.type == 'video')

    ### get number of frames ###
    nFrames = streamObj.frames

    ### get ??? idk what the ticksPerFrame really is --> ask Jeff ###
    ticksPerFrame = streamObj.rate / streamObj.time_base
    width = streamObj.width

    ### return video object dictionary with all the elements ###
    vidObjDict = {'avObj':avObj, 'streamObj':streamObj,
                  'nFrames':nFrames, 'ticksPerFrame':ticksPerFrame, 'type':'raw',
                  'width':streamObj.width, 'height':streamObj.height}


    return vidObjDict, avObj, streamObj, ticksPerFrame  #  avObj, streamObj, nFrames, ticksPerFrame


######################################################################################################
######################################################################################################
#### this function will be to skip forward in the video to show first frame with needed fixation #####

def skip_forward_to_first_desired_frame(vidObj, firstFixationFrame, currentFrame):

    ### loop through frames within the frames of interest ###
    ### want to start at the first fixation and move forward ###
    while currentFrame < firstFixationFrame:

        ### keep skipping forward through frames until I reach the frame I want ###
        if (firstFixationFrame - currentFrame) > 100:
            print("Jumping ahead to firstFixationFrame.  currentFrame = {} firstFixationFrame = {} (jumping to {})".format(currentFrame, firstFixationFrame, firstFixationFrame-9))

            ### Speed things up by jumping to the next frame ###
            ### jump forward to almost the frame you want. ###
            vidObj.set(1, (firstFixationFrame-50))

            ### update counter ###
            currentFrame = firstFixationFrame-50

            ### grab the current frame ###
            (grabbed, frame) = vidObj.read()

        ### iterature through loop ###
        currentFrame = currentFrame + 1

    return None

######################################################################################################
######################################################################################################
############ this function will be for grabbing the needed frame in the raw video file ###############

def grab_video_frame(avObj, streamObj, vidObjDict, ticksPerFrame, frameNumToRead=0):

    ### define stream object ###
    streamObj = vidObjDict['streamObj']

    ### make sure video is a raw video file ###
    if vidObjDict['type'] == 'raw':

        ### With raw mp4, we can go directly to the next fixation frame instead of stepping through each frame ...###
        ### initialize to default return value ###
        frame = None

        if frameNumToRead is not None:  # if a frame# is given, seek to it
            # To seek to a specific frame number:
            frameNumToRead = max(frameNumToRead,0)  # don't allow negative frame numbers
            frameNumToRead = min(frameNumToRead,streamObj.frames-2)  # don't allow frame numbers > nFrames

            seekTicks = int(frameNumToRead * ticksPerFrame)
            streamObj.seek(seekTicks)
        else:  # read the next available frame
            frameNumToRead = -1  # Flag to indicate 'take next'

        for packet in avObj.demux(streamObj):  # even if I only want one frame, I have to demux bc of compression
            for frame in packet.decode():
                frameNum = packet.pts / ticksPerFrame

                if frameNumToRead == -1:
                    frameNumToRead = frameNum

                npImg = np.array(np.frombuffer(frame.to_rgb().planes[0].to_bytes(), dtype = np.uint8)).reshape(streamObj.height, streamObj.width, 3)
            frame = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)

            if frameNum == frameNumToRead:  # Got it; return
                break
    ### using converted video files with buffer ###
    else:
        ### Grab the next frame (from buffer, or file if needed) ###
        (grabbed, frame) = vidObjDict['vidObj'].read()

    vidObjDict['currentFrame'] = frameNumToRead

    return frame, vidObjDict

##############################################################################################################
##############################################################################################################
####################################### initial feature detection ############################################

def feature_detect(frame, frameMask, ref_img, method=DETECTOR, matcher=MATCHER):

    #src_pts = 0
    #dst_pts = 0
    ### choose which one when calling function; by default, method is SIFT ###

    if method == "SIFT":
        det = cv2.xfeatures2d.SIFT_create()

    elif method == "SURF" :
        det = cv2.xfeatures2d.SURF_create()

    elif method == "ORB":
        det = cv2.ORB_create()

    ### get keypoints and descriptors ###
    kp1, des1 = det.detectAndCompute(frame, frameMask)
    kp2, des2 = det.detectAndCompute(ref_img, None)

    ### SIFT and SURF incorporate same method for feature detection ###
    if matcher == "FLANN":

        ####Need to use a FLann matcher to find matches
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        ### get good matches ###
        ### store all the good matches as per Lowe's ratio test. ###
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        ### Get src and dst points ###
        ### use try/except because for some frames, not enough kpts will be detected ###
        ### will get UnboundLocalError, so use try/except to move around that ###
        if len(good_matches) > 0:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        else:
            print("Not enough matches to get source and distance points...")
            # src_pts = ([0,0])
            # dst_pts = ([0,0])
            frame += 1


    ### ORB will incorporate different detection method ###
    elif matcher == "BF":

        ##BFMatcher with default parameters
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        ### get src and dst points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)


    ### create homography matrix ###
    # try:
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    return dst_pts, H, mask, good_matches, kp1, kp2

##############################################################################################################
##############################################################################################################
############# this function will be the MAIN one to complete the object detection in the frames ##############

def object_detect(ref_img, dst_pts, segW, segH, nRowsImg, nColsImg, mask):

    ### squeeze dst pts to make a (67,2) array
    dst_pts = np.squeeze(dst_pts, axis=1)

    ### multiple dst pts by the mask to get the coordinate points in the reference image
    coords = dst_pts * mask

    ### want to find all coordinates that aren't 0,0. ###
    ### iterate through coordinates and add to the total_nonzero count if coordinates aren't 0,0 ###
    total_nonzero = 0
    for (x,y) in coords:
        if (x,y) == (0,0):
            continue
        else:
            total_nonzero += 1

            ### sum up the coordinates and set equal to the total x and y ###
            x_total, y_total = np.sum(coords, axis = 0)

            ### height and width of ref image ###
            ref_img_height, ref_img_width = ref_img.shape[:2]

            ### may need to define the number of rows and columns in ref image corresponding to the number of objects within ###
            ### if 3x2 objects, set rows to 2 and columns to 3
            ref_img_rows = nRowsImg
            ref_img_cols = nColsImg

            ### define the height and width of the object to be the total ref img height/width divided by the rows and columns ###
            ### if ref img height = 240 snd there are 4 rows, obj height will be 60 ###
            obj_height = int(ref_img_height/ref_img_rows)
            obj_width = int(ref_img_width/ref_img_cols)

            ### find the avg coordinate by dividing the total sum by the number of nonzero coordinates ###
            ### basically finding single coordinate point that averages out all coordinates ###
            ### location will be where highest concentration of coordinates are, ergo the best matching region ###
            x_avg = x_total/total_nonzero
            y_avg = y_total/total_nonzero

            ### create index x and y points
            index_x = int(x_avg/obj_width)
            index_y = int(y_avg/obj_height)

            ### multiple indexes by the obj height and width to make index be size of each object ###
            ### change in index moves one object at a time rather than one pixel at a time ###
            index_x = int((index_x) * obj_width)
            index_y = int((index_y) * obj_height)

    ### return these four coords to draw a rectangle in display_image function ###
    return index_x, index_y, obj_height, obj_width, x_avg, y_avg

############################################################################################################
############################################################################################################
######### This part of code draws matches and does perspective transform for feature matching ###############

def perspectiveTransform(file, ref_img, kp1, kp2, good, matches, matchesMask, mask, M, method=DETECTOR):

    height,width = file.shape[:2]
    pts = np.float32([[0,0], [0,height-1], [width-1,height-1], [width-1,0]]).reshape(-1,1,2)
    ### do a perspective transform
    dst = cv2.perspectiveTransform(pts,M)

    img3 = cv2.polylines(ref_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (255,0,0), singlePointColor = None,
        matchesMask = matchesMask,flags=2)

    ### Draw a bounding box around the best match in the reference image ###
    ### for SIFT or SURF, need to use 'good' matches ###
    ### for ORB, just use matches because not enough keypoints for detection if good matches is done ###

    if method == "SIFT" or method == "SURF":
        img3 = cv2.drawMatches(file, kp1, ref_img, kp2, good, None, **draw_params)

    elif method == "ORB":
        img3 = cv2.drawMatches(file, kp1, ref_img, kp2, matches, None, **draw_params)

    textScale = 1.25
    cv2.putText(img3, '{} matches'.format(len(matches)), (3, 150), FONT, textScale, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img3, 'detection method = {}'.format(method), (5, int(100*textScale)), FONT, 1.0*textScale, (255,255,255), 1, cv2.LINE_AA)

    return img3

##############################################################################################################
##############################################################################################################
################### this function is used to stick two images next to each other horizontally ################

def hstack(img_left, img_right, border=0):
    # horizontally concatenate two images. If different depths, make both color

    ### separate function to horizontally concatenate images together ###
    def add_border(img_left, borderImg):

        ### define shape ###
        shapeL, shapeBorder = img_left.shape, borderImg.shape

        ### make sure images have same bit depths ###
        if len(shapeL) != len(shapeBorder):
            ### if images grayscale, convert to BGR ###
            if len(shapeL) == 2:
                img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
            if len(shapeBorder) == 2:  # shapeBorder is grayscale; convert it to BGR
                borderImg = cv2.cvtColor(borderImg, cv2.COLOR_GRAY2BGR)

        ### height and width of both images ###
        hL, wL = img_left.shape[:2]
        hBorder, wBorder = borderImg.shape[:2]

        ### color image ###
        if img_left.shape[2] == 3:

            ### create empty matrix ###
            ### image with border ###
            imgWborder = np.zeros((max(hL, hBorder), (wL + wBorder), 3), np.uint8)

            ### combine 2 images ###
            ### create new window based on height and width of both images ###
            imgWborder[0:hL, 0:wL, 0:3] = img_left
            imgWborder[0:hBorder, wL:wL+wBorder, 0:3] = borderImg

        ### if monochrome image ###
        else:
            ### create empty matrix ###
            ### image with border ###
            imgWborder = np.zeros(((hL + hBorder), max(wL, wBorder)), np.uint8)

            ### combine 2 images ###
            ### don't include depth of images ###
            imgWborder[0:hL, 0:wL] = img_left
            imgWborder[0:hBorder, wL:wL+wBorder] = borderImg

        ### return image with border ###
        return imgWborder

    ### define shape, height, and width of left anf right images ###
    shapeL, shapeR = img_left.shape, img_right.shape
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]

    ### Add a border to the bottom of img_top the same width as img_top and border pixels high ###
    if border > 0:

        ### make it grayscale; it will be converted if necessary ###
        borderImg = np.zeros((hL, border), np.uint8)

        ### add the border to the bottom of img_top, then continue ... ###
        img_left = add_border(img_left, borderImg)

    ### Recalculate shape, height, and width now that border is added ###
    shapeL, shapeR = img_left.shape, img_right.shape
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]

    ### again with the various bit depths ###
    ### img_left is grayscale; convert it to BGR ###
    ### img_right is grayscale; convert it to BGR ###
    if len(shapeL) != len(shapeR):
        if len(shapeL) == 2:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        if len(shapeR) == 2:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

    ### height and width of left and right images ###
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]

    ### create final stack image ###
    ### if color image: ###
    if len(img_left.shape) == 3:
        ### create empty matrix size of that max height and width of both images + border ###
        stackImg = np.zeros((max(hL, hR), wL + wR, 3), np.uint8)

        ### combine 2 images ###
        ### take left side as left image, from that end to end of window as right image ###
        stackImg[:hL, :wL, :3] = img_left
        stackImg[:hR, wL:wL + wR, :3] = img_right

    ### repeat for monochrome image ###
    else:
        ### create empty matrix ###
        stackImg = np.zeros((max(hL, hR), wL + wR), np.uint8)

        ### combine 2 images ###
        stackImg[:hL, :wL] = img_left
        stackImg[:hL, wL:wL + wR] = img_right

    ### return stack image ###
    return stackImg

##############################################################################################################
##############################################################################################################
################# this function is used to stick two images/windows next to each other verically #############

def vstack_images(img_top, img_bottom, border=0):
    # vertically concatenate two images. If different depths, make both color
    # optionally, add a border between them

    def add_border(img_top, borderImg):
        # vertically concatenate one image with a border.

        shapeT, shapeBorder = img_top.shape, borderImg.shape

        if len(shapeT) != len(shapeBorder):  # Images are different bit depths
            if len(shapeT) == 2:  # img_top is grayscale; convert it to BGR
                img_top = cv2.cvtColor(img_top, cv2.COLOR_GRAY2BGR)
            if len(shapeBorder) == 2:  # shapeBorder is grayscale; convert it to BGR
                borderImg = cv2.cvtColor(borderImg, cv2.COLOR_GRAY2BGR)

        hT, wT = img_top.shape[:2]
        hBorder, wBorder = borderImg.shape[:2]

        if img_top.shape[2] == 3:  # color image
            # create empty matrix
            imgWborder = np.zeros(((hT + hBorder), max(wT, wBorder), 3), np.uint8)

            # combine 2 images
            imgWborder[0:hT, 0:wT, 0:3] = img_top
            imgWborder[hT:hT + hBorder, 0:wBorder, 0:3] = borderImg

        else:  # monochrome image
            # create empty matrix
            # stackImg = np.zeros((max(hT, hB), wT + wB, 2), np.uint8)
            imgWborder = np.zeros(((hT + hBorder), max(wT, wBorder), 2), np.uint8)

            # combine 2 images
            imgWborder[0:hT, 0:wT, 0:2] = img_top
            imgWborder[hT:hT + hBorder, 0:wBorder, 0:2] = borderImg

        return imgWborder

    # vertically concatenate two images. If different depths, make both color
    # optionally, add a border.
    shapeT, shapeB = img_top.shape, img_bottom.shape
    hT, wT = img_top.shape[:2]
    hB, wB = img_bottom.shape[:2]

    if border > 0:  # Add a border to the bottom of img_top the same width as img_top and border pixels high
        borderImg = np.zeros((border, wT), np.uint8)  # make it grayscale; it will be converted if nec.

        img_top = add_border(img_top, borderImg)  # add the border to the bottom of img_top, then continue ...

    # Recalculate now that border is added
    shapeT, shapeB = img_top.shape, img_bottom.shape
    hT, wT = img_top.shape[:2]
    hB, wB = img_bottom.shape[:2]

    if len(shapeT) != len(shapeB):  # Images are different bit depths
        if len(shapeT) == 2:  # img_top is grayscale; convert it to BGR
            img_top = cv2.cvtColor(img_top, cv2.COLOR_GRAY2BGR)
        if len(shapeB) == 2:  # img_bottom is grayscale; convert it to BGR
            img_bottom = cv2.cvtColor(img_bottom, cv2.COLOR_GRAY2BGR)

    if img_top.shape[2] == 3:  # color image
        # create empty matrix
        # stackImg = np.zeros((max(hT, hB), wT + wB, 3), np.uint8)
        stackImg = np.zeros(((hT + hB), max(wT, wB), 3), np.uint8)

        # combine 2 images
        stackImg[0:hT, 0:wT, 0:3] = img_top
        stackImg[hT:hT+hB, 0:wB, 0:3] = img_bottom

    else: # monochrome image
        # create empty matrix
        # stackImg = np.zeros((max(hT, hB), wT + wB, 2), np.uint8)
        stackImg = np.zeros(((hT + hB), max(wT, wB), 2), np.uint8)

        # combine 2 images
        stackImg[0:hT, 0:wT, 0:2] = img_top
        stackImg[hT:hT+hB, 0:wB, 0:2] = img_bottom

    return stackImg


######################################################################################################
######################################################################################################

def object_display_image(ref_img, test_img,fixRegionImg, fixPosXY, index_x, index_y, obj_height, obj_width, currentFrame,
                                fixTableIdx, H, mask, good_matches, kp1, kp2):

    ### define x and y coordinates of the fixation ###
    fix_x = fixPosXY[0]
    fix_y = fixPosXY[1]

    ### concatenate a 1 to the end of the coordinate to be able to multiply by 3x3 Homography Matrix ###
    fixPosXY = np.concatenate((fixPosXY, 1), axis=None)
    # print('frame fixation = {}'.format(fixPosXY))

    ### use try/except to move past errors when there is no homography matrix for frame ###
    ### errors in homography matrix can occur when fixation isn't on the object, so not enough kpts for matrix ###
    try:
        ### get fixation coordinate on reference image by multiplying by mask from homography ###
        ref_fix = np.dot(fixPosXY, H.T)

    except Exception as errMsg:
        ref_fix = np.array([850,850,1])
        print('No Banknote found for frame {}; fixation positioned outside references at ({})'.format(currentFrame, ref_fix))

    ref_fix = ref_fix/ref_fix[2]

    # print('ref fixation = {}'.format(ref_fix))
    ref_fix_x = int(ref_fix[0])
    ref_fix_y = int(ref_fix[1])

    ### draw circle around the initial index coordinate (optional) ###
    result = cv2.circle(test_img, (fix_x, fix_y), 20, YELLOW, 3)

    ### create rectangle starting at index point and making second point be index + obj height/width ###
    result = cv2.rectangle(ref_img, (index_x, index_y), (index_x + obj_width, index_y+obj_height), GREEN, 6)

    ### draw circle around fixation in reference image region ###
    result = cv2.circle(ref_img, (ref_fix_x, ref_fix_y), 20, BLUE, 3)

    ### vstack fixRegion with the frame to create left side of image ###
    left_img = vstack_images(test_img, fixRegionImg, border=2)

    ### hstack final image with the frame/fixRegion combo ###
    displayImg = hstack(left_img, result, border=2)

    # draw_params = dict(matchColor = (255,0,0), singlePointColor = None,
    #     matchesMask = mask,flags=2)
    #
    # displayImg = cv2.drawMatchesKnn(fixRegionImg, kp1, ref_img, kp2, good_matches, None, **draw_params)

    ### Text to add to image window ###
    frame_text = "Current frame: {}".format(currentFrame)
    fix_text = "Current Fixation: {}".format(fixTableIdx)
    fix_pos_text = "Fixation Position: ({},{})".format(fixPosXY[0], fixPosXY[1])
    ref_fix_text = "Fix Pos in Ref Frame: ({},{})".format(ref_fix_x, ref_fix_y)
    matches_text = "Good Matches: {}".format(len(good_matches))

    ### display the text over the final display window ###
    cv2.putText(displayImg, frame_text, (4, 1000), FONT, 4, CYAN, 2, cv2.LINE_AA)
    cv2.putText(displayImg, fix_text, (4, 1150), FONT, 2.5, WHITE, 2, cv2.LINE_AA)
    cv2.putText(displayImg, fix_pos_text, (1, 1200), FONT, 2.5, WHITE, 2, cv2.LINE_AA)
    cv2.putText(displayImg, ref_fix_text, (2100, 950), FONT, 2, MAGENTA, 2, cv2.LINE_AA)
    cv2.putText(displayImg, matches_text, (4, 1250), FONT, 2.5, YELLOW , 1, cv2.LINE_AA)


    ### create a display image name for each frame/image ###
    displayImg_name = ('displayImg_{}-'.format(test_img))

    ### display each resulting window ###
    # cv2.imshow(displayImg_name, displayImg)

    ### waitKey ###
    # k = cv2.waitKey(1)  # display
    # if (k & 0xff) == 27 or (k & 0xff) == 113 or (k & 0xff) == 81: #if Esc, q or Q is pushed
    #     k = 'exit' #exit program
    #     print('exiting...')
    #     sys.exit()

    return ref_fix, displayImg

######################################################################################################
######################################################################################################

def write_stringlist_to_csv_file(fName, headerTxt, outputStringlist):
    csvFile = open(fName,'w')
    csvFile.write(headerTxt + '\n')

    for csvStr in outputStringlist:
        csvFile.write(csvStr + '\n')
    csvFile.close()


##############################################################################################################
##############################################################################################################

def append_stringlist_to_csv_file(fName, outputString):

    # Append one line of text to existing csv file and close it.
    csvFile = open(fName,'a')

    csvFile.write(outputString + '\n')

    csvFile.close()

######################################################################################################
########### Now begins the process to calculate distance to ROI and visualizing statistics ##########
######################################################################################################

######################################################################################################
######################################################################################################

######################################################################################################
######################################################################################################

######################################################################################################
######################################################################################################

######################################################################################################
######################################################################################################

##############################################################################################################
##############################################################################################################
############################## object detection for the test images ##########################################

def object_detect_test(test_img, ref_img, dst_pts, mask):

    ### squeeze dst pts to make a (67,2) array
    dst_pts = np.squeeze(dst_pts, axis=1)

    ### multiple dst pts by the mask to get the coordinate points in the reference image
    coords = dst_pts * mask

    ### want to find all coordinates that aren't 0,0. ###
    ### iterate through coordinates and add to the total_nonzero count if coordinates aren't 0,0 ###
    total_nonzero = 0
    for (x,y) in coords:
        if (x,y) == (0,0):
            continue
        else:
            total_nonzero += 1

            ### sum up the coordinates and set equal to the total x and y ###
            x_total, y_total = np.sum(coords, axis = 0)

            ### height and width of ref image ###
            ref_img_height, ref_img_width = ref_img.shape[:2]

            ### may need to define the number of rows and columns in ref image corresponding to the number of objects within ###
            ### if 3x2 objects, set rows to 2 and columns to 3
            ref_img_rows = 2
            ref_img_cols = 3

            ### define the height and width of the object to be the total ref img height/width divided by the rows and columns ###
            ### if ref img height = 240 snd there are 4 rows, obj height will be 60 ###
            obj_height = int(ref_img_height/ref_img_rows)
            obj_width = int(ref_img_width/ref_img_cols)

            ### find the avg coordinate by dividing the total sum by the number of nonzero coordinates ###
            ### basically finding single coordinate point that averages out all coordinates ###
            ### location will be where highest concentration of coordinates are, ergo the best matching region ###
            x_avg = x_total/total_nonzero
            y_avg = y_total/total_nonzero

            ### create index x and y points
            index_x = int(x_avg/obj_width)
            index_y = int(y_avg/obj_height)

            ### multiple indexes by the obj height and width to make index be size of each object ###
            ### change in index moves one object at a time rather than one pixel at a time ###
            index_x = int((index_x) * obj_width)
            index_y = int((index_y) * obj_height)

            ### draw circle around the initial index coordinate (optional) ###
            img3 = cv2.circle(ref_img, (x_avg, y_avg), 10, WHITE, 3)

            ### create rectangle starting at index point and making second point be index + obj height/width ###
            img3 = cv2.rectangle(ref_img, (index_x, index_y), (index_x + obj_width, index_y+obj_height), GREEN, 8)


            return img3

######################################################################################################
######################################################################################################