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

    if method == "SIFT" or method == "SURF":
        good = matches[:10]
        src_pts = numpy.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    elif method == "ORB":
        src_pts = numpy.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    #get good matches
    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
        # if m.distance < 0.7*n.distance:
            # good.append(m)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    height,width = img1.shape[:2]
    pts = numpy.float32([[0,0], [0,height-1], [width-1,height-1], [width-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[numpy.int32(dst)],True,255,3, cv2.LINE_AA)


    draw_params = dict(matchColor = (255,0,0), singlePointColor = None,
                        matchesMask = matchesMask,flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)



    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)


    return bf, matches, img3


# def find_best_match(matches):

    # top_match = list(matches)[-1]
    # second_match = list(matches)[-2]


if __name__ == "__main__":

    from matplotlib import pyplot as plt


    img1 = cv2.imread('IMG_8196.JPG') #test image
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread('6_mag_window_2_dark.png') #ref image
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    bf, matches, detImg = feature_detect(img1, img2, "ORB")

    cv2.namedWindow('Matches')
    cv2.imshow('Matches', detImg)

    cv2.waitKey()

    cv2.imwrite('6_mag_result.png', detImg)
    cv2.destroyAllWindows()


    # plt.imshow(detImg, 'gray')
    # plt.show()
