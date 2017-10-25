import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, [0, 0, 0], [1, 0, 0], [2, 0, 0],...,[7, 5, 0]
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].transpose().reshape(-1, 2)

# arrays to store object points and image points
objpoints = []
imgpoints = []

images = glob.glob("/home/lsq/CarND-term1/Udacity-CarND-term1/P2-advanced-lane-lines/"
                   "calibration_wide/GOPR00*.jpg")
for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board cornes
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # if found, add object points, image points
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

