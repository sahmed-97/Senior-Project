import cv2
import numpy as np

DETECTOR = 'SIFT'

MATCHER = 'FLANN'

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

##############################################################################################################
##############################################################################################################

def feature_detect(test_img, ref_img, method=DETECTOR, matcher=MATCHER):


###various feature detection methods...

#######choose which one when calling function; by default, method is SIFT######

    if method == "SIFT":
        det = cv2.xfeatures2d.SIFT_create()

    elif method == "SURF" :
        det = cv2.xfeatures2d.SURF_create()

    elif method == "ORB":
        det = cv2.ORB_create()

    kp1, des1 = det.detectAndCompute(test_img, None)
    kp2, des2 = det.detectAndCompute(ref_img, None)

    ### SIFT and SURF incorporate same method for feature detection ###
    if matcher == "FLANN":

        ####Need to use a FLann matcher to find matches
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        ## get good matches
        ## store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        #good = matches[:10]

        ### Get src and dst points
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

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
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    return src_pts, dst_pts, M, mask

##############################################################################################################
##############################################################################################################

def object_detect(test_img, ref_img, dst_pts, mask):

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

##############################################################################################################
##############################################################################################################

def object_detection_fixation(test_img, ref_image, fixation_xy):

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

    ### define x and y positions from the given fixation from the table ###
    x_pos = fixation_xy[0]
    y_pos = fixation_xy[1]

    ### create index x and y points
    index_x = int(x_pos/obj_width)
    index_y = int(y_pos/obj_height)

    ### multiple indexes by the obj height and width to make index be size of each object ###
    ### change in index moves one object at a time rather than one pixel at a time ###
    index_x = int((index_x) * obj_width)
    index_y = int((index_y) * obj_height)

    ### draw circle around the initial index coordinate (optional) ###
    img3 = cv2.circle(ref_img, (index_x, index_y), 10, WHITE, 3)

    ### create rectangle starting at index point and making second point be index + obj height/width ###
    img3 = cv2.rectangle(ref_img, (index_x, index_y), (index_x + obj_width, index_y+obj_height), GREEN, 6)


    return img3


    ######### This part of code draws matches and does perspective transform for feature matching ###############
def perspectiveTransform(test_img, ref_img, mask, M):

    height,width = file.shape[:2]
    pts = np.float32([[0,0], [0,height-1], [width-1,height-1], [width-1,0]]).reshape(-1,1,2)
    ### do a perspective transform
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(ref_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

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
def hstack(imgL, imgR, border=0, verbose=False):
    # horizontally concatenate two images. If different depths, make both color

    def add_border(imgL, borderImg, verbose=False):
        # horizontally concatenate one image with a border.

        shapeL, shapeBorder = imgL.shape, borderImg.shape
        if verbose:
            print("shapeL = {}".format(shapeL))

        if len(shapeL) != len(shapeBorder):  # Images are different bit depths
            if len(shapeL) == 2:  # imgT is grayscale; convert it to BGR
                imgT = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
            if len(shapeBorder) == 2:  # shapeBorder is grayscale; convert it to BGR
                borderImg = cv2.cvtColor(borderImg, cv2.COLOR_GRAY2BGR)

        hL, wL = imgL.shape[:2]
        hBorder, wBorder = borderImg.shape[:2]

        if imgL.shape[2] == 3:  # color image
            # create empty matrix
            imgWborder = np.zeros((max(hL, hBorder), (wL + wBorder), 3), np.uint8)
                      #  np.zeros((max(hL, hR),       wL + wR, 3), np.uint8)

            # combine 2 images
            imgWborder[0:hL, 0:wL, 0:3] = imgL
            imgWborder[0:hBorder, wL:wL+wBorder, 0:3] = borderImg
            # stackImg[0:hL, 0:wL, 0:3] = imgL
            # stackImg[0:hR,      wL:wL + wR, 0:3] = imgR

        else:  # monochrome image
            # create empty matrix
            # stackImg = np.zeros((max(hT, hB), wT + wB, 2), np.uint8)
            imgWborder = np.zeros(((hL + hBorder), max(wL, wBorder)), np.uint8)

            # combine 2 images
            imgWborder[0:hL, 0:wL] = imgL
            imgWborder[0:hBorder, wL:wL+wBorder] = borderImg

        return imgWborder

    shapeL, shapeR = imgL.shape, imgR.shape
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]

    if border > 0:  # Add a border to the bottom of imgT the same width as imgT and border pixels high
        borderImg = np.zeros((hL, border), np.uint8)  # make it grayscale; it will be converted if nec.
        imgL = add_border(imgL, borderImg)  # add the border to the bottom of imgT, then continue ...

    # Recalculate now that border is added
    shapeL, shapeR = imgL.shape, imgR.shape
    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]

    if len(shapeL) != len(shapeR):  # Images are different bit depths
        if len(shapeL) == 2:  # imgL is grayscale; convert it to BGR
            imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        if len(shapeR) == 2:  # imgR is grayscale; convert it to BGR
            imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    hL, wL = imgL.shape[:2]
    hR, wR = imgR.shape[:2]


    if len(imgL.shape) == 3:  # color image
        # create empty matrix
        stackImg = np.zeros((max(hL, hR), wL + wR, 3), np.uint8)

        # combine 2 images
        stackImg[:hL, :wL, :3] = imgL
        stackImg[:hR, wL:wL + wR, :3] = imgR

    else: # monochrome image
        # create empty matrix
        stackImg = np.zeros((max(hL, hR), wL + wR), np.uint8)

        # combine 2 images
        stackImg[:hL, :wL] = imgL
        stackImg[:hL, wL:wL + wR] = imgR

    return stackImg

######################################################################################################
######################################################################################################

##############################################################################################################
##############################################################################################################
def vstack_images(imgT, imgB, border=0):
    # vertically concatenate two images. If different depths, make both color
    # optionally, add a border between them

    def add_border(imgT, borderImg):
        # vertically concatenate one image with a border.

        shapeT, shapeBorder = imgT.shape, borderImg.shape

        if len(shapeT) != len(shapeBorder):  # Images are different bit depths
            if len(shapeT) == 2:  # imgT is grayscale; convert it to BGR
                imgT = cv2.cvtColor(imgT, cv2.COLOR_GRAY2BGR)
            if len(shapeBorder) == 2:  # shapeBorder is grayscale; convert it to BGR
                borderImg = cv2.cvtColor(borderImg, cv2.COLOR_GRAY2BGR)

        hT, wT = imgT.shape[:2]
        hBorder, wBorder = borderImg.shape[:2]

        if imgT.shape[2] == 3:  # color image
            # create empty matrix
            imgWborder = np.zeros(((hT + hBorder), max(wT, wBorder), 3), np.uint8)

            # combine 2 images
            imgWborder[0:hT, 0:wT, 0:3] = imgT
            imgWborder[hT:hT + hBorder, 0:wBorder, 0:3] = borderImg

        else:  # monochrome image
            # create empty matrix
            # stackImg = np.zeros((max(hT, hB), wT + wB, 2), np.uint8)
            imgWborder = np.zeros(((hT + hBorder), max(wT, wBorder), 2), np.uint8)

            # combine 2 images
            imgWborder[0:hT, 0:wT, 0:2] = imgT
            imgWborder[hT:hT + hBorder, 0:wBorder, 0:2] = borderImg

        return imgWborder

    # vertically concatenate two images. If different depths, make both color
    # optionally, add a border.

    shapeT, shapeB = imgT.shape, imgB.shape
    hT, wT = imgT.shape[:2]
    hB, wB = imgB.shape[:2]

    if border > 0:  # Add a border to the bottom of imgT the same width as imgT and border pixels high
        borderImg = np.zeros((border, wT), np.uint8)  # make it grayscale; it will be converted if nec.

        imgT = add_border(imgT, borderImg)  # add the border to the bottom of imgT, then continue ...

    # Recalculate now that border is added
    shapeT, shapeB = imgT.shape, imgB.shape
    hT, wT = imgT.shape[:2]
    hB, wB = imgB.shape[:2]

    if len(shapeT) != len(shapeB):  # Images are different bit depths
        if len(shapeT) == 2:  # imgT is grayscale; convert it to BGR
            imgT = cv2.cvtColor(imgT, cv2.COLOR_GRAY2BGR)
        if len(shapeB) == 2:  # imgB is grayscale; convert it to BGR
            imgB = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    if imgT.shape[2] == 3:  # color image
        # create empty matrix
        # stackImg = np.zeros((max(hT, hB), wT + wB, 3), np.uint8)
        stackImg = np.zeros(((hT + hB), max(wT, wB), 3), np.uint8)

        # combine 2 images
        stackImg[0:hT, 0:wT, 0:3] = imgT
        stackImg[hT:hT+hB, 0:wB, 0:3] = imgB

    else: # monochrome image
        # create empty matrix
        # stackImg = np.zeros((max(hT, hB), wT + wB, 2), np.uint8)
        stackImg = np.zeros(((hT + hB), max(wT, wB), 2), np.uint8)

        # combine 2 images
        stackImg[0:hT, 0:wT, 0:2] = imgT
        stackImg[hT:hT+hB, 0:wB, 0:2] = imgB

    return stackImg

##############################################################################################################
##############################################################################################################
