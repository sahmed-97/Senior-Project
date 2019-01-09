import cv2
import numpy

def feature_detect(img1, img2, method='SIFT'):


#if verbose is true...
    # if verbose == True:
        # msg = "could not complete action"
        # raiseError(msg)

###various feature detection methods...
###choose which one when calling function; by default, method is SIFT

    if method == "SIFT":
        det = cv2.xfeatures2d.SIFT_create()

    elif method == "SURF" :
        det = cv2.xfeatures2d.SURF_create()

    elif method == "ORB":
        det = cv2.ORB_create()


    kp1, des1 = det.detectAndCompute(img1, None)
    kp2, des2 = det.detectAndCompute(img2, None)

    ##BFMatcher with default parameters
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    #get good matches
    good_matches = matches[:10]


    # Use the best 10 matches to form a transformation matrix
    src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good_matches     ]).reshape(-1,1,2)
    dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    #find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape[:2]
    pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    #Transform the rectangle around img1 based on the transformation matrix
    dst = cv2.perspectiveTransform(pts,M)
    dst += (w, 0)  # adding offset to put bounding box at correct positon

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

    # Draw bounding box in Red
    img3 = cv2.polylines(img3, [numpy.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)




    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)


    return bf, matches, img3


# def find_best_match(matches):

    # top_match = list(matches)[-1]
    # second_match = list(matches)[-2]


if __name__ == "__main__":

    img1 = cv2.imread('mints.JPG')
    img2 = cv2.imread('setting2.JPG')

    bf, matches, detImg = feature_detect(img1, img2, "ORB")

    cv2.namedWindow('Matches')
    cv2.imshow('Matches', detImg)

    cv2.waitKey()

    cv2.imwrite('mints_results.png', detImg)
    cv2.destroyAllWindows()
