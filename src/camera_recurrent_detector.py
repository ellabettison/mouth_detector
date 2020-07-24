from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from image_pre_processor import get_mouth_pixels_without_preprocessing, pre_process_image

last_five_frames = np.empty((0, 256), int)
suppression_delay = 0


def predict(inp):
    return model.predict(inp)


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

model = load_model("../rnn_model.h5")

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    mouth_pixels = get_mouth_pixels_without_preprocessing(frame)
    processed_frame = pre_process_image(mouth_pixels)
    last_five_frames = np.append(last_five_frames, np.array([processed_frame]).reshape((1, 256)), axis=0)
    if len(last_five_frames) > 5:
        last_five_frames = np.delete(last_five_frames, 0, axis=0)
        play = predict(last_five_frames.reshape((1, 5, 256)))
        # print(play)
        if play > 0.5:
            suppression_delay = 10
        if suppression_delay > 0:
            frame = cv.rectangle(frame, (0, 0), (50, 50), (255, 0, 0), 5)
        suppression_delay -= 1
    cv.imshow("frame", frame)
    if cv.waitKey(10) == 27:
        break
