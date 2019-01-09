import cv2
import numpy
# import math


def feature_detection(img1, img2, method = "SIFT", verbose=False):

    if verbose == True:
        msg = "could not complete action"
        raiseError(msg)



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

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:-1], None, flags=2)



    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags = 2)

    # best_match = min(matches)

    return bf, matches, img3



if __name__ == "__main__":


    img1 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/Original_panda.png')
    img2 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/window_test3.png')

    # img1 = cv2.imread('watch_resize.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread('magazine_window2.png')

    # img1 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame0.jpg')
    # img2 = cv2.imread('C:/Users/sheel/Desktop/4th Year/Senior Project/frame2594.jpg')


    bf, matches, detImg = feature_detection(img1, img2, "ORB")

    cv2.namedWindow('Matches')
    cv2.imshow('Matches', detImg)

    cv2.waitKey()

    cv2.imwrite('mag_results.png', detImg)
    cv2.destroyAllWindows()
