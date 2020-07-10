# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
import argparse
from collections import OrderedDict

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68))
])

LEFT_EDGE = 48
RIGHT_EDGE = 54
TOP_TOP_MIDDLE = 51
TOP_BOTTOM_MIDDLE = 62
BOTTOM_TOP_MIDDLE = 66
BOTTOM_BOTTOM_MIDDLE = 57

important_vals = [LEFT_EDGE, RIGHT_EDGE, TOP_TOP_MIDDLE,
                  TOP_BOTTOM_MIDDLE, BOTTOM_TOP_MIDDLE,
                  BOTTOM_BOTTOM_MIDDLE]

def display_points(shape):
    # clone the original image so we can draw on it, then
    # display the name of the face part on the image
    clone = image.copy()
    cv2.putText(clone, "moth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    # loop over the subset of facial landmarks, drawing the
    # specific face part

    i = 48
    for (x, y) in shape[48:68]:
        if i in important_vals:
            cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
        else:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        i += 1

    # extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

    # show the particular face part
    cv2.imshow("ROI", roi)
    cv2.imshow("Image", clone)
    cv2.waitKey(0)

def get_mouth(args, gray):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    rect = rects[0]

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    display_points(shape)



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

get_mouth(args, gray)
