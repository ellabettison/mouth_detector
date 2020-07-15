from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import tensorflow as tf

from neural_net_mouth_detector import get_mouth_pixels, predict

long_norm = None
norm = None
last_five_frames = np.array([0, 0, 0])


def detectAndDisplay(frame):
    pixels = get_mouth_pixels(frame)
    if pixels is not None:
        pixels = np.array(pixels).reshape((-1, 255, 1))
        pixels = pixels.astype(np.float)
        pixels = tf.convert_to_tensor(pixels)
        prob = predict(pixels)
        print("PROBABILITY: ", prob)
        frame = cv.putText(frame, str(prob), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                           cv.LINE_AA)
        if prob[0][0] > 0.9:
            frame = cv.rectangle(frame, (0, 0), (50, 50), (255, 0, 0), 5)

        cv.imshow('Capture - Face detection', frame)


# np.set_printoptions(precision=3)
#
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='../haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
#
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break